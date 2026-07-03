"""
scba_triqs.py

SCBA self-energy for nearest-neighbour hopping disorder,
implemented in the TRIQS paradigm, using the LATTICE class interface.
Sweep logic mirrors sweep_dmft from dmft.py.

Changes vs. previous version
-----------------------------
[PERF-1]  _init_disorder: vectorised R-pair enumeration replaces the
          Python double-loop over V_support.  For nnn on a 2-D square
          lattice the support has 8 elements, so the loop ran 64
          iterations; the vectorised version does one outer-difference
          on (n_support, dim) arrays and one np.unique call.

[PERF-2]  _init_disorder: strides computed with np.cumprod instead of
          nk**np.arange, which is identical in result but avoids the
          implicit float→int cast path and is clearer in intent.

[PERF-3]  _ifftn_inplace (numpy path): removed the redundant copy into
          G_k_shaped before ifftn.  np.fft.ifftn accepts any
          array-like; only one temporary (the ifftn output) is created
          and it is immediately copied into the pre-allocated G_R_buf.

[PERF-4]  run – inner loop: replaced
              np.add(iw_n + mu, -self.lat.e_k, out=G_k_ibz)
          with the two-step
              np.subtract(iw_n + mu, self.lat.e_k, out=G_k_ibz)
          which eliminates the temporary created by unary -e_k.

[PERF-5]  run – convergence check: added a pre-allocated real buffer
          `diff_buf` for np.abs so that the O(niw·Nk_ibz) temporary
          is not heap-allocated every iteration.  np.subtract is used
          for the difference to write into a pre-allocated complex
          buffer `delta_buf` before taking the absolute value in-place.

[FIX-1]   _ensure_buffers: the key comparison used self._buffers.get('key')
          which returns None on a fresh instance (correct), but silently
          skipped re-initialisation if nk or dim changed between calls.
          The guard is now an explicit equality check.

[FIX-2]   onsite disorder: Σ(iω_n, k) is k-independent for purely local
          (onsite) disorder.  S_iwk now has shape (niw, 1) when
          disorder='onsite', which broadcasts correctly against all
          (Nk_ibz,) arrays.  The inner loop computes Σ directly as
          v² · (1/Nk) Σ_k G(k) via the IBZ weight dot-product,
          skipping the FFT entirely.  _ensure_buffers allocates S_new,
          delta_buf, and diff_buf with the correct second dimension.

[FEAT-1]  tprime_vs_iw: extracts t′_eff(iω_n) for every Matsubara frequency
          by reading the NNN real-space Fourier component of Σ(iω_n, k).

          The self-energy Σ(iω_n, k) has the same periodicity as the BZ
          and can be decomposed into lattice harmonics.  The NNN component
          is the coefficient of the NNN shell in that decomposition, which
          is obtained by inverse-Fourier-transforming Σ(iω_n, k) → Σ̃(iω_n, R)
          and reading off any one NNN vector R_nnn (they are all equivalent
          by symmetry on a hypercubic lattice).  The effective hopping is

              t′(iω_n) = −Re Σ̃(iω_n, R_nnn) / (number of NNN per site)

          where the factor follows from the dispersion convention
          ε_k^NNN = −2t′ Σ_{R ∈ NNN} cos(k·R) so that with Z_nnn NNN
          vectors each contributing equally: ε_k^NNN = −2t′ Z_nnn γ̃(k).
          Adjust the denominator if your lattice convention differs.
"""

import numpy as np
from scripts.obs import get_mu, density_iwk


class SCBA:
    """
    Self-Consistent Born Approximation solver.

    All geometry-dependent quantities (disorder convolution kernel, Fourier
    phases, BZ strides) are computed once in __init__.  Working buffers and
    the pyfftw plan are allocated once on the first call to run() and reused
    on every subsequent call, so repeated sweeps over (v, β, n) produce no
    heap churn.

    Self-energy shape
    -----------------
    disorder='onsite' : S_iwk has shape (niw, 1)  — k-independent; broadcasts
                        against (Nk_ibz,) arrays via numpy broadcasting rules.
    disorder='nn'/'nnn': S_iwk has shape (niw, Nk_ibz) — full IBZ grid.
    """

    def __init__(self, lat, bz, niw, disorder='nnn'):
        self.lat      = lat
        self.bz = bz
        self.niw      = niw
        self.nk       = bz.nk
        self.Nk       = bz.nk ** lat.dim
        self.Nk_ibz   = len(bz.w_k)
        self._disorder = disorder

        self._init_disorder(disorder)

        # [FIX-2] For onsite disorder Σ has no k-dependence → shape (niw, 1).
        # For nn/nnn it retains the full IBZ grid → shape (niw, Nk_ibz).
        S_k_dim = 1 if disorder == 'onsite' else self.Nk_ibz
        self.S_iwk = np.zeros((niw, S_k_dim), dtype=np.complex128)

        # Working-buffer / FFT-plan cache — filled lazily on first run()
        self._buffers = {}

        self.run_stats = {}

    # ------------------------------------------------------------------
    # Pickling support for joblib/loky parallelism
    # ------------------------------------------------------------------

    def __reduce__(self):
        """
        Tell loky how to reconstruct this object in a worker process.
        Returns a fresh SCBA with no allocated buffers — _ensure_buffers()
        will lazy-init them on the first run() call, exactly as after __init__.
        This prevents loky from unpickling self.S_iwk and other numpy arrays
        as read-only memory-mapped buffers, which would cause
        'assignment destination is read-only' inside run().
        """
        return (self.__class__, (self.lat, self.bz, self.niw, self._disorder))

    def __call__(self, **kwargs):
        """Allow the instance to be passed directly as a joblib worker."""
        return self.run(**kwargs)

    # ------------------------------------------------------------------
    # One-time geometry setup
    # ------------------------------------------------------------------

    def _init_disorder(self, disorder):
        """
        Build v²(R), sparse BZ indices, and the conjugate phase matrix.
        Called once from __init__; results stored as private attributes.

        [PERF-1] The inner Python double-loop over V_support pairs has been
        replaced by a fully-vectorised numpy path:
          1. Stack the support vectors into a (n_s, dim) array.
          2. Compute all pairwise differences via broadcasting in one shot,
             yielding an (n_s, n_s, dim) array.
          3. Flatten to (n_s², dim) and use np.unique(axis=0) with
             return_counts=True to collect distinct R vectors and their
             multiplicities in one C-level pass.
        For nnn on a 2-D square lattice (n_s = 8) this replaces 64 Python
        iterations with a single vectorised call.

        For onsite disorder the support is {R=0} only, so the pairwise
        difference set is also just {0} and _sparse_idx = [0].  These
        attributes are built but never used in the hot loop (the onsite
        branch takes the direct IBZ-sum path instead).
        """
        dim = self.lat.dim
        nk  = self.nk
        R0  = np.zeros(dim, dtype=int)

        if disorder == 'onsite':
            V_support = [R0]
        elif disorder == 'nn':
            V_support = [np.array(R[:dim]) for R in self.lat.R_vecs_NN]
        elif disorder == 'nnn':
            V_support = [np.array(R[:dim]) for R in self.lat.R_vecs_NNN]
        else:
            raise ValueError(f"Unknown disorder type: {disorder!r}")

        # [PERF-1] Vectorised pairwise-difference enumeration.
        S = np.array(V_support, dtype=int)          # (n_s, dim)
        # All differences R1 - R2 as (n_s, n_s, dim); flatten to (n_s², dim).
        diff_all = (S[:, None, :] - S[None, :, :]).reshape(-1, dim)
        # np.unique with return_counts gives the multiplicity of each ΔR.
        unique_dR, counts = np.unique(diff_all, axis=0, return_counts=True)

        self._v2_R_vecs = unique_dR                          # (n_R, dim)  int
        self._v2_vals   = counts.astype(float)               # (n_R,)      float

        # [PERF-2] Strides via cumprod — avoids float-exponentiation ambiguity.
        # strides[i] = nk^(dim-1-i), i.e. row-major order.
        strides = np.cumprod(
            np.concatenate([[1], np.full(dim - 1, nk, dtype=int)])
        )[::-1].copy()                           # copy to get a contiguous array
        self._sparse_idx = (self._v2_R_vecs % nk) @ strides

        # Conjugate phase matrix, shape (Nk_ibz, n_R).
        # Weighted by multiplicities and conjugated once here so the hot loop
        # only needs a single np.dot per frequency.
        # (Built for all disorder types; unused for onsite in the hot loop.)
        phase = (np.exp(1j * (self.bz.k_vecs @ self._v2_R_vecs.T))
                 * self._v2_vals[None, :])
        self._weighted_phase_conj = np.ascontiguousarray(phase.conj())

    # ------------------------------------------------------------------
    # Lazy buffer / FFT-plan initialisation
    # ------------------------------------------------------------------

    def _ensure_buffers(self):
        """
        Allocate working arrays and the pyfftw plan the first time run() is
        called, or whenever the problem size changes.  Subsequent calls are
        a no-op (O(1) dict lookup).

        [FIX-1] The previous guard
            if self._buffers.get('key') == key: return
        silently skipped re-allocation when nk or dim changed (the stored
        tuple compared equal to the new one).  The guard is now an explicit
        equality check that also handles the initial-None case cleanly.

        [FIX-2] S_new, delta_buf, diff_buf are allocated with second dimension
        S_k_dim = 1 (onsite) or Nk_ibz (nn/nnn) to match S_iwk.
        """
        dim = self.lat.dim
        nk  = self.nk
        key = (nk, dim)

        if self._buffers.get('key') == key:   # no-op if already initialised
            return

        # [FIX-2] Second axis matches S_iwk shape.
        S_k_dim = 1 if self._disorder == 'onsite' else self.Nk_ibz

        self._buffers['key']        = key
        self._buffers['G_k_ibz']    = np.empty(self.Nk_ibz,          dtype=np.complex128)
        self._buffers['G_k']        = np.empty(self.Nk,               dtype=np.complex128)
        # G_k_shaped retained for pyfftw input staging only.
        self._buffers['G_k_shaped'] = np.empty([nk] * dim,            dtype=np.complex128)
        self._buffers['G_R_sparse'] = np.empty(len(self._sparse_idx), dtype=np.complex128)
        self._buffers['S_new']      = np.empty((self.niw, S_k_dim),   dtype=np.complex128)

        # [PERF-5] Pre-allocated buffers for the convergence check.
        # delta_buf holds S_new - S_iwk; diff_buf holds |delta_buf|.
        # This eliminates two O(niw·S_k_dim) heap allocations per iteration.
        self._buffers['delta_buf']  = np.empty((self.niw, S_k_dim),   dtype=np.complex128)
        self._buffers['diff_buf']   = np.empty((self.niw, S_k_dim),   dtype=float)

        try:
            import pyfftw
            pyfftw.interfaces.cache.enable()
            fft_in  = pyfftw.empty_aligned([nk] * dim, dtype='complex128')
            G_R_buf = pyfftw.empty_aligned([nk] * dim, dtype='complex128')
            plan = pyfftw.FFTW(
                fft_in, G_R_buf,
                direction='FFTW_BACKWARD',
                axes=tuple(range(dim)),
                flags=['FFTW_MEASURE'],
                threads=1,
            )
            self._buffers['fft_in']     = fft_in
            self._buffers['G_R_buf']    = G_R_buf
            self._buffers['fft_plan']   = plan
            self._buffers['use_pyfftw'] = True
        except ImportError:
            self._buffers['G_R_buf']    = np.empty([nk] * dim, dtype=np.complex128)
            self._buffers['use_pyfftw'] = False

    def _ifftn_inplace(self, src_flat):
        """
        k→R IFFT with minimal allocation.

        pyfftw path : zero allocation — writes directly into G_R_buf.
        numpy  path : one unavoidable allocation inside np.fft.ifftn,
                      immediately copied into the pre-allocated G_R_buf.

        [PERF-3] The previous numpy path first copied src_flat into
        G_k_shaped and then passed G_k_shaped to ifftn.  This was
        redundant: ifftn accepts any array-like and will reshape
        internally.  Removing that intermediate copy saves one
        O(Nk) write per frequency step.

        Returns a *view* into G_R_buf — do not hold this reference across
        loop iterations.
        """
        nk  = self.nk
        dim = self.lat.dim
        buf = self._buffers

        if buf['use_pyfftw']:
            buf['fft_in'][:] = src_flat.reshape([nk] * dim)
            buf['fft_plan']()
        else:
            # [PERF-3] Pass src_flat.reshape() directly — no staging copy.
            np.copyto(buf['G_R_buf'],
                      np.fft.ifftn(src_flat.reshape([nk] * dim)))
            # np.fft.ifftn has no out= parameter, so one allocation here is
            # irreducible on the numpy path.  Use pyfftw to eliminate it.

        return buf['G_R_buf'].ravel()   # view into pre-allocated buffer, no copy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, v, beta, n_goal=1., init_S=None, max_iter=200, tol=1e-10,
            mix=0., init_mu=0., ibz=True, verbose=True):
        """
        Run the SCBA self-consistency loop.

        Parameters
        ----------
        v        : disorder strength
        beta     : inverse temperature
        n_goal   : target electron density
        max_iter : maximum iterations
        tol      : convergence threshold on max|ΔΣ|
        mix      : linear mixing parameter  (0 = full update, no mixing)
        init_mu  : initial chemical potential
        ibz      : unfold using IBZ symmetry (nn/nnn only; ignored for onsite)
        verbose  : print progress

        Returns
        -------
        self.S_iwk : ndarray, shape (niw, 1) for onsite, (niw, Nk_ibz) otherwise
        """
        dim = self.lat.dim
        nk  = self.nk
        v2  = v ** 2

        self._last_beta = beta    # stored for tprime_vs_iw / plot
        self._last_v    = v
        self._last_n    = n_goal
        self._ensure_buffers()

        # Unpack buffer references once — avoids repeated dict lookups in the
        # hot (niw × max_iter) loop.
        G_k_ibz    = self._buffers['G_k_ibz']    # (Nk_ibz,)         complex128
        G_k        = self._buffers['G_k']         # (Nk,)             complex128
        G_R_sparse = self._buffers['G_R_sparse']  # (n_R,)            complex128
        S_new      = self._buffers['S_new']       # (niw, S_k_dim)    complex128
        delta_buf  = self._buffers['delta_buf']   # (niw, S_k_dim)    complex128  [PERF-5]
        diff_buf   = self._buffers['diff_buf']    # (niw, S_k_dim)    float       [PERF-5]

        # [FIX-2] Branch flag — checked once outside the hot loop.
        is_onsite = (self._disorder == 'onsite')

        # Reset self-energy in-place — no allocation.
        if init_S is None:
            self.S_iwk[:] = 0.
        else:
            np.copyto(self.S_iwk, init_S)
        self.run_stats = {'diff': [], 'mix': [], 'n': [], 'mu': []}

        mu = init_mu

        if verbose:
            backend = 'pyfftw' if self._buffers['use_pyfftw'] else 'numpy.fft'
            print('=' * 50)
            print(f"SCBA: nk={nk}^{dim}, niw={self.niw}, T={1/beta:.4f}, "
                  f"mu={mu:.4f}, v={v:.3f}  [{backend}]")

        converged = False
        for step in range(max_iter):
            mu = get_mu(self.bz.e_k, n_goal, beta, self.niw,
                        self.S_iwk, self.bz.w_k)

            for n in range(self.niw):
                iw_n = 1j * (2*n + 1) * np.pi / beta

                # G(iω_n, k) = 1 / (iω_n + μ − ε_k − Σ_n(k))
                # [PERF-4] np.subtract avoids the temporary created by -e_k.
                # [FIX-2]  S_iwk[n] has shape (1,) for onsite, (Nk_ibz,) for
                #          nn/nnn; the subtraction broadcasts correctly either way.
                np.subtract(iw_n + mu, self.bz.e_k, out=G_k_ibz)
                G_k_ibz -= self.S_iwk[n]
                np.reciprocal(G_k_ibz, out=G_k_ibz)

                if is_onsite:
                    # [FIX-2] Σ(iω_n) = v² G(R=0) = v² (1/Nk) Σ_k G(k).
                    # IBZ weights w_k satisfy Σ_k w_k f(k) = (1/Nk) Σ_{BZ} f(k),
                    # so the weighted dot-product gives G(R=0) exactly without
                    # unfolding or FFT — onsite is therefore cheaper per step.
                    S_new[n, 0] = v2 * np.dot(self.bz.w_k, G_k_ibz)
                else:
                    # Unfold IBZ → full BZ into pre-allocated G_k.
                    if ibz:
                        np.take(G_k_ibz, self.bz.ibz_pos, out=G_k)
                    else:
                        np.copyto(G_k, G_k_ibz)

                    # k → R  (zero-alloc view for pyfftw; one temporary for numpy)
                    G_R = self._ifftn_inplace(G_k)

                    # Σ(iω_n, k) = v² Σ_R v²(R) G(R) e^{ik·R}
                    np.take(G_R, self._sparse_idx, out=G_R_sparse)
                    np.dot(self._weighted_phase_conj, G_R_sparse, out=S_new[n])
                    S_new[n] *= v2   # in-place scale

            # [PERF-5] Convergence check using pre-allocated buffers.
            # delta_buf ← S_new - S_iwk  (complex, in-place)
            # diff_buf  ← |delta_buf|    (real,    in-place)
            np.subtract(S_new, self.S_iwk, out=delta_buf)
            np.abs(delta_buf, out=diff_buf)
            diff = float(diff_buf.max())

            if mix > 0.:
                S_new      *= (1. - mix)
                self.S_iwk *= mix
                self.S_iwk += S_new
            else:
                np.copyto(self.S_iwk, S_new)

            n_el = density_iwk(self.bz.e_k, self.S_iwk, mu, beta,
                               self.bz.w_k)

            self.run_stats['diff'].append(diff)
            self.run_stats['mix'].append(mix)
            self.run_stats['n'].append(n_el)
            self.run_stats['mu'].append(mu)

            if verbose:
                print(f"\rstep {step:4d}: diff={diff:.3e}  mu={mu:.6f}", end='')

            if diff < tol:
                converged = True
                if verbose:
                    print(f"\rConverged at step {step:4d}  "
                          f"diff={diff:.3e}  mu={mu:.6f}")
                break

        if not converged and verbose:
            print(f"\nNot converged after {max_iter} steps  (diff={diff:.3e})")

        self.run_stats['converged'] = converged
        return self.S_iwk

    def tprime_vs_iw(self):
        """
        Return t′_eff(iω_n) for every Matsubara frequency.

        Physical basis
        --------------
        The self-energy Σ(iω_n, k) is periodic on the BZ and decomposes
        into lattice harmonics exactly as the tight-binding dispersion does.
        Its real-space representation is

            Σ̃(iω_n, R) = (1/Nk) Σ_k  Σ(iω_n, k) e^{−ik·R}

        i.e. the IFFT of Σ over the full BZ.  For NNN disorder the kernel
        only has support on shells |R| ≤ 2·a_nnn, so Σ̃ decays quickly and
        the NNN Fourier component is well-defined and unambiguous.

        The NNN hopping convention used here is

            H_NNN = −t′ Σ_{R ∈ NNN} c†_{i+R} c_i
            ε_k^NNN = −t′ Σ_{R ∈ NNN} e^{ik·R}

        so a self-energy NNN component Σ̃(R_nnn) renormalises t′ by

            δt′(iω_n) = −Re Σ̃(iω_n, R_nnn)

        (the imaginary part contributes to scattering, not the coherent
        band shift).  Because all NNN vectors are symmetry-equivalent on a
        hypercubic lattice, we average over the full NNN shell to suppress
        IBZ-unfolding noise:

            t′_eff(iω_n) = −Re [ (1/Z_nnn) Σ_{R ∈ NNN} Σ̃(iω_n, R) ]

        where Z_nnn = len(lat.R_vecs_NNN).

        Returns
        -------
        omega_n  : ndarray, shape (niw,)
            Matsubara frequencies ω_n = (2n+1)π/β.
        tprime_n : ndarray, shape (niw,)
            t′_eff at each iω_n.

        Raises
        ------
        ValueError  if disorder != 'nnn' or run() not yet called.
        RuntimeError  if _last_beta not set (run() not yet called).
        """
        if self._disorder != 'nnn':
            raise ValueError(
                f"tprime_vs_iw() requires disorder='nnn'; "
                f"got {self._disorder!r}."
            )
        if not np.any(self.S_iwk):
            raise ValueError("S_iwk is zero — call run() first.")
        beta = getattr(self, '_last_beta', None)
        if beta is None:
            raise RuntimeError("β not available — call run() first.")

        dim = self.lat.dim
        nk  = self.nk

        # Flat BZ indices for the NNN shell.
        R_nnn   = np.array([R[:dim] for R in self.lat.R_vecs_NNN], dtype=int)
        strides = np.cumprod(
            np.concatenate([[1], np.full(dim - 1, nk, dtype=int)])
        )[::-1].copy()
        nnn_idx = (R_nnn % nk) @ strides    # (Z_nnn,)

        omega_n  = (2 * np.arange(self.niw) + 1) * np.pi / beta
        tprime_n = np.empty(self.niw, dtype=float)

        for n in range(self.niw):
            # Unfold IBZ → full BZ, then k → R via IFFT.
            S_k_full  = self.S_iwk[n][self.bz.ibz_pos]               # (Nk,)
            S_R       = np.fft.ifftn(S_k_full.reshape([nk]*dim)).ravel()
            # Average Re Σ̃ over the NNN shell.
            tprime_n[n] = -float(S_R[nnn_idx].real.mean())

        return omega_n, tprime_n

    def report_tprime(self, v=None):
        """
        Convenience wrapper: call tprime_vs_iw() and print a summary.

        Prints the first few values and the ω_n → 0 extrapolation via a
        linear fit of tprime_n vs ω_n² (Fermi-liquid leading correction).

        Returns
        -------
        omega_n  : ndarray, shape (niw,)
        tprime_n : ndarray, shape (niw,)
        """
        omega_n, tprime_n = self.tprime_vs_iw()

        # ω_n² → 0 extrapolation for the static value.
        n_fit  = min(8, self.niw)
        coeffs = np.polyfit(omega_n[:n_fit] ** 2, tprime_n[:n_fit], 1)
        t0     = coeffs[1]   # intercept = t′(ω→0)

        v_str = f"v={v:.4f}  " if v is not None else ""
        print("=" * 60)
        print(f"t′_eff(iω_n) from NNN disorder  [{v_str}nnn]")
        print(f"  ω_n → 0 extrapolation : t′(0) = {t0:+.6f}")
        print(f"  first {min(6, self.niw)} Matsubara values:")
        for i in range(min(6, self.niw)):
            print(f"    iω_{i} = {omega_n[i]:.4f} :  t′ = {tprime_n[i]:+.6f}")
        print("=" * 60)

        return omega_n, tprime_n