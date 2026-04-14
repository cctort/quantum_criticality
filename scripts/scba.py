"""
scba_hopping_disorder.py

SCBA self-energy for nearest-neighbour hopping disorder on a cubic lattice.

Physics
-------
Random hopping δt_ij with zero mean, variance W^2 (per bond).
Disorder correlator in momentum space:
    C(k, k') = (W/t)^2 * γ_{k-k'}^2
where γ_q = (1/d) Σ_α cos(q_α)  (structure factor of NN bonds).

SCBA self-consistency:
    Σ(k, iω_n) = (W^2 / N) Σ_{k'} γ_{k-k'}^2 * G(k', iω_n)
    G(k, iω_n) = 1 / (iω_n + μ - ε_k - Σ(k, iω_n))

Two implementations are provided:

1. scba_hopping_disorder      — FFT-based convolution, O(Nk^d log Nk · Nω).
                                 Exact, used as cross-check.

2. scba_hopping_disorder_fast — 3-moment reduction, O(Nk^d · Nω).
                                 Exploits the exact decomposition of γ_{k-k'}^2
                                 derived from the cosine addition formula:

   γ_{k-k'}^2 = (1/d^2)[B1(k)·cos²(k'_x) + B2(k)·cos(k'_x)cos(k'_y) + B3(k)·sin²(k'_x)]

   where:
       B1(k)  = Σ_α cos²(k_α)
       B2(k)  = 2·Σ_{α<β} cos(k_α)cos(k_β)
       B3(k)  = Σ_α sin²(k_α)  =  d - B1(k)

   The self-energy lives in span{B1, B2, B3} — an A1g-symmetric subspace
   of the second-harmonic sector — and the self-consistency closes in
   only 3 complex scalars per Matsubara frequency.

   The k-dependence of Σ is physically significant: the scattering rate
   is large at zone-face centres X=(π,0,0) and small at Γ and M=(π,π,π).
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------

def make_kmesh(Nk: int, d: int = 3) -> np.ndarray:
    """
    Returns (Nk^d, d) array of k-points in [0, 2π).
    """
    ks = np.linspace(0, 2 * np.pi, Nk, endpoint=False)
    grids = np.meshgrid(*([ks] * d), indexing='ij')
    return np.stack([g.ravel() for g in grids], axis=-1)


def dispersion(kvecs: np.ndarray, t: float = 1.0) -> np.ndarray:
    """ε_k = -2t Σ_α cos(k_α),  shape (Nk^d,)"""
    return -2 * t * np.cos(kvecs).sum(axis=-1)


def gamma_k(kvecs: np.ndarray) -> np.ndarray:
    """γ_k = (1/d) Σ_α cos(k_α) ∈ [-1, 1],  shape (Nk^d,)"""
    return np.cos(kvecs).mean(axis=-1)


def matsubara_frequencies(n_max: int, beta: float) -> np.ndarray:
    """Fermionic Matsubara frequencies iω_n for n = -n_max … n_max-1."""
    ns = np.arange(-n_max, n_max)
    return np.pi / beta * (2 * ns + 1)


# ---------------------------------------------------------------------------
# FFT-based implementation
# ---------------------------------------------------------------------------

def scba_hopping_disorder(
    Nk: int,
    beta: float,
    mu: float,
    W: float,
    t: float = 1.0,
    d: int = 3,
    n_matsubara: int = 256,
    max_iter: int = 200,
    tol: float = 1e-7,
    mix: float = 0.4,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SCBA for NN hopping disorder via FFT convolution.

    Computes Σ(k, iω_n) = (W/t)^2 · (1/N) Σ_{k'} γ_{k-k'}^2 G(k', iω_n)
    as a circular convolution using FFT at each Matsubara frequency.

    Parameters
    ----------
    Nk          : k-points per dimension.
    beta        : Inverse temperature.
    mu          : Chemical potential.
    W           : Disorder strength (std-dev of δt, same units as t).
    t           : Hopping amplitude.
    d           : Spatial dimension (2 or 3).
    n_matsubara : Half the number of Matsubara frequencies (total = 2*n_matsubara).
    max_iter    : Max SCBA iterations.
    tol         : Convergence threshold on max|ΔΣ|.
    mix         : Linear mixing parameter (0 < mix ≤ 1).
    verbose     : Print convergence info.

    Returns
    -------
    sigma : (Nk^d, 2*n_matsubara) complex  —  Σ(k, iω_n)
    G     : (Nk^d, 2*n_matsubara) complex  —  G(k, iω_n)
    iw    : (2*n_matsubara,) real           —  Matsubara frequencies
    """
    assert d in (2, 3), "Only d=2,3 supported"
    grid_shape = (Nk,) * d
    N = Nk ** d

    kvecs  = make_kmesh(Nk, d)
    eps_k  = dispersion(kvecs, t)
    gam2_k = gamma_k(kvecs) ** 2
    gam2_grid = gam2_k.reshape(grid_shape)

    iw = matsubara_frequencies(n_matsubara, beta)
    Nw = len(iw)

    sigma = np.zeros((N, Nw), dtype=complex)
    pf    = (W / t) ** 2

    if verbose:
        print(f"SCBA FFT: Nk={Nk}, d={d}, β={beta:.2f}, μ={mu:.4f}, "
              f"W={W:.3f}, t={t:.3f}")
        print(f"  {N} k-points, {Nw} Matsubara frequencies")

    F_gam2 = np.fft.fftn(gam2_grid)   # precompute — real, so only once

    for iteration in range(max_iter):
        G = 1.0 / (1j * iw[None, :] + mu - eps_k[:, None] - sigma)

        sigma_new = np.zeros_like(sigma)
        for wi in range(Nw):
            G_grid = G[:, wi].reshape(grid_shape)
            conv   = np.fft.ifftn(F_gam2 * np.fft.fftn(G_grid)) / N
            sigma_new[:, wi] = pf * conv.ravel()

        err   = np.max(np.abs(sigma_new - sigma))
        sigma = sigma + mix * (sigma_new - sigma)

        if verbose and (iteration % 10 == 0 or err < tol):
            print(f"  iter {iteration:4d}  |ΔΣ|_max = {err:.3e}")

        if err < tol:
            if verbose:
                print(f"  Converged in {iteration+1} iterations.")
            break
    else:
        if verbose:
            print(f"  Warning: did not converge in {max_iter} iterations "
                  f"(|ΔΣ|_max = {err:.3e})")

    G = 1.0 / (1j * iw[None, :] + mu - eps_k[:, None] - sigma)
    return sigma, G, iw


# ---------------------------------------------------------------------------
# Fast 3-moment implementation
# ---------------------------------------------------------------------------

def scba_hopping_disorder_fast(
    Nk: int,
    beta: float,
    mu: float,
    W: float,
    t: float = 1.0,
    d: int = 3,
    n_matsubara: int = 256,
    max_iter: int = 300,
    tol: float = 1e-8,
    mix: float = 0.3,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast SCBA using the exact 3-moment decomposition of γ_{k-k'}^2.

    From the cosine addition formula:
        γ_{k-k'} = (1/d) Σ_α [cos(k_α)cos(k'_α) + sin(k_α)sin(k'_α)]

    Squaring and averaging over k' with weight G(k'), only 3 independent
    moments survive (sin(k'_α)sin(k'_β) cross terms with α≠β vanish by
    parity since G depends only on ε_k = -2t Σ cos(k_α)):

        Σ(k, iω) = (W/t)²/d² · [B1(k)·Mcc + B2(k)·Mccxy + B3(k)·Mss]

    Basis functions (k-dependent, fixed throughout):
        B1(k)  = Σ_α cos²(k_α)
        B2(k)  = 2·Σ_{α<β} cos(k_α)cos(k_β)
        B3(k)  = Σ_α sin²(k_α)  =  d - B1(k)

    Moments (3 complex scalars per Matsubara frequency, iterated):
        Mcc   = (1/N) Σ_{k'} cos²(k'_x) G(k', iω)
        Mccxy = (1/N) Σ_{k'} cos(k'_x)cos(k'_y) G(k', iω)
        Mss   = (1/N) Σ_{k'} sin²(k'_x) G(k', iω)

    Cost: O(Nk^d · Nω) per iteration. Matches FFT variant exactly.

    The k-dependence of Σ is substantial:
        Γ=(0,0,0)  and  M=(π,π,π): B1=3, B2=6, B3=0  →  low scattering
        X=(π,0,0)              :    B1=1, B2=-2, B3=2  →  high scattering
    """
    N     = Nk ** d
    kvecs = make_kmesh(Nk, d)
    eps_k = dispersion(kvecs, t)

    cx = np.cos(kvecs[:, 0])
    cy = np.cos(kvecs[:, 1])
    cz = np.cos(kvecs[:, 2]) if d == 3 else np.zeros(N)
    sx = np.sin(kvecs[:, 0])

    if d == 3:
        B1 = cx**2 + cy**2 + cz**2
        B2 = 2 * (cx*cy + cx*cz + cy*cz)
    else:
        B1 = cx**2 + cy**2
        B2 = 2 * cx * cy
    B3 = d - B1   # = Σ_α sin²(k_α)

    iw = matsubara_frequencies(n_matsubara, beta)
    Nw = len(iw)
    pf = (W / t) ** 2 / d ** 2

    Mcc   = np.zeros(Nw, dtype=complex)
    Mccxy = np.zeros(Nw, dtype=complex)
    Mss   = np.zeros(Nw, dtype=complex)

    if verbose:
        print(f"SCBA fast (3-moment): Nk={Nk}, d={d}, β={beta:.2f}, "
              f"μ={mu:.4f}, W={W:.3f}, t={t:.3f}")

    for iteration in range(max_iter):
        sigma_k = pf * (
            B1[:, None] * Mcc[None, :]
          + B2[:, None] * Mccxy[None, :]
          + B3[:, None] * Mss[None, :]
        )
        G = 1.0 / (1j * iw[None, :] + mu - eps_k[:, None] - sigma_k)

        Mcc_new   = (cx[:, None]**2         * G).mean(axis=0)
        Mccxy_new = (cx[:, None]*cy[:, None] * G).mean(axis=0)
        Mss_new   = (sx[:, None]**2         * G).mean(axis=0)

        err = max(np.max(np.abs(Mcc_new   - Mcc)),
                  np.max(np.abs(Mccxy_new - Mccxy)),
                  np.max(np.abs(Mss_new   - Mss)))

        Mcc   = (1 - mix) * Mcc   + mix * Mcc_new
        Mccxy = (1 - mix) * Mccxy + mix * Mccxy_new
        Mss   = (1 - mix) * Mss   + mix * Mss_new

        if verbose and (iteration % 20 == 0 or err < tol):
            print(f"  iter {iteration:4d}  |ΔM|_max = {err:.3e}")

        if err < tol:
            if verbose:
                print(f"  Converged in {iteration+1} iterations.")
            break
    else:
        if verbose:
            print(f"  Warning: not converged ({max_iter} iters), |ΔM|={err:.3e}")

    sigma_k = pf * (
        B1[:, None] * Mcc[None, :]
      + B2[:, None] * Mccxy[None, :]
      + B3[:, None] * Mss[None, :]
    )
    G = 1.0 / (1j * iw[None, :] + mu - eps_k[:, None] - sigma_k)
    return sigma_k, G, iw


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def filling_from_G(G: np.ndarray, beta: float) -> float:
    """
    Compute filling n (spin-summed, ∈ [0,2]) from G(k, iω_n).
    Uses: n_k = (1/β) Σ_n Re[G(k, iω_n)] + 1/2  (leading-order Matsubara sum).
    Factor of 2 for spin.
    """
    n_k = G.real.sum(axis=1) / beta + 0.5
    return float(2 * n_k.mean())


def scba_sigma_reshaped(sigma: np.ndarray, Nk: int, d: int = 3) -> np.ndarray:
    """Reshape sigma from (Nk^d, Nw) → (Nk, ..., Nk, Nw) for use in chi0_at_q."""
    Nw = sigma.shape[1]
    return sigma.reshape((Nk,) * d + (Nw,))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Nk   = 16
    beta = 10.0
    mu   = 0.0
    t    = 1.0
    d    = 3

    # Index helpers
    def k_idx(kx, ky, kz):
        return kx * Nk**2 + ky * Nk + kz

    Gamma_idx = k_idx(0,      0,      0)
    X_idx     = k_idx(Nk//2,  0,      0)
    M_idx     = k_idx(Nk//2,  Nk//2,  Nk//2)

    print("=== Testing full FFT-based SCBA ===")
    for W in [0.0, 0.5, 1.0]:
        sigma, G, iw = scba_hopping_disorder(
            Nk=Nk, beta=beta, mu=mu, W=W, t=t, d=d,
            n_matsubara=128, max_iter=200, tol=1e-6, mix=0.5, verbose=False
        )
        n     = filling_from_G(G, beta)
        n_pos = sigma.shape[1] // 2
        print(f"  W={W:.1f}: n={n:.3f}, "
              f"-Im Σ(Γ)={-sigma[Gamma_idx, n_pos].imag:.4f}, "
              f"-Im Σ(X)={-sigma[X_idx,     n_pos].imag:.4f}, "
              f"-Im Σ(M)={-sigma[M_idx,     n_pos].imag:.4f}")

    print("\n=== Testing fast 3-moment SCBA ===")
    for W in [0.0, 0.5, 1.0, 1.5]:
        sigma, G, iw = scba_hopping_disorder_fast(
            Nk=Nk, beta=beta, mu=mu, W=W, t=t, d=d,
            n_matsubara=128, max_iter=300, tol=1e-8, mix=0.3, verbose=False
        )
        n     = filling_from_G(G, beta)
        n_pos = sigma.shape[1] // 2
        print(f"  W={W:.1f}: n={n:.3f}, "
              f"-Im Σ(Γ)={-sigma[Gamma_idx, n_pos].imag:.4f}, "
              f"-Im Σ(X)={-sigma[X_idx,     n_pos].imag:.4f}, "
              f"-Im Σ(M)={-sigma[M_idx,     n_pos].imag:.4f}")

    print("\nDone.")