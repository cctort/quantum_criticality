import itertools
import numpy as np
from numba import njit, prange

class TetraQuadrature:
    def __init__(self, n_quad=6):
        self.lam, self.qw = build_tetra_quadrature(n_quad)
        self.tets = kuhn_tetrahedra_3d()

def gauss01(n):
    """n-point Gauss-Legendre nodes/weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


def build_tetra_quadrature(n_quad=6):
    """
    Collapsed-coordinate (Duffy transform) Gauss-Legendre quadrature on the
    reference tetrahedron with vertices v0..v3.

    Returns
    -------
    lam : (n_nodes, 4) float64
        Barycentric weights (l0,l1,l2,l3) at each node; point = sum_i li*vi.
    qw  : (n_nodes,) float64
        Quadrature weights. sum(qw) == 1/6 (volume of the reference simplex).

    n_quad=5 or 6 is a good default (validated against independent fine-grid
    and Monte Carlo references). Since nodes only ever touch the cheap
    closed-form kernel (not e_k itself), raising n_quad is nearly free in
    terms of new physics evaluations -- just more arithmetic.
    """
    u, wu = gauss01(n_quad)
    v, wv = gauss01(n_quad)
    w, ww = gauss01(n_quad)
    U, V, W = np.meshgrid(u, v, w, indexing='ij')
    WU, WV, WW = np.meshgrid(wu, wv, ww, indexing='ij')

    lam1 = U.ravel()
    lam2 = ((1 - U) * V).ravel()
    lam3 = ((1 - U) * (1 - V) * W).ravel()
    lam0 = ((1 - U) * (1 - V) * (1 - W)).ravel()
    weight = (WU * WV * WW * (1 - U) ** 2 * (1 - V)).ravel()

    return np.stack([lam0, lam1, lam2, lam3], axis=1), weight


def kuhn_tetrahedra_3d():
    """
    Kuhn (body-diagonal) decomposition of the unit cube into 6 tetrahedra.
    Returns int64 array (6,4,3) of 0/1 corner offsets, one row of 4 vertices
    per tetrahedron, tiling the cube exactly (each tet volume = 1/6 cube).
    """
    tets = []
    for perm in itertools.permutations([0, 1, 2]):
        v = np.zeros((4, 3), dtype=np.int64)
        cur = np.zeros(3, dtype=np.int64)
        for step, axis in enumerate(perm):
            cur = cur.copy()
            cur[axis] = 1
            v[step + 1] = cur
        tets.append(v)
    return np.array(tets)

@njit(inline='always')
def _Fermi(z):
    if z > 500.:
        return 0.
    elif z < -500.:
        return 1.
    else:
        return 1. / (np.exp(z) + 1.)


@njit(parallel=True, cache=True)
def lindhard_tetra_3d(mu, beta, e_k_grid, e_kq_grid, tet_offsets, lam, qw):
    """
    Tetrahedron-method static Lindhard chi0, same normalization as lindhard_ksp
    (i.e. same physical quantity, drop-in numerically-superior replacement).

    Parameters
    ----------
    e_k_grid, e_kq_grid : (nk,nk,nk) float64
        FULL periodic BZ grid (unfold IBZ first if needed). Same layout as
        e_k.reshape((nk,)*dim) used elsewhere in lattice.py.
    tet_offsets : (6,4,3) int64   -- from kuhn_tetrahedra_3d()
    lam, qw     :                 -- from build_tetra_quadrature()

    Returns
    -------
    chi0 : float64
    """
    nk = e_k_grid.shape[0]
    Nk = nk * nk * nk
    n_nodes = lam.shape[0]
    n_tet = tet_offsets.shape[0]

    partial = np.zeros(nk, dtype=np.float64)

    for i in prange(nk):
        s = 0.0
        for j in range(nk):
            for k in range(nk):
                a_corner = np.empty(8, dtype=np.float64)
                b_corner = np.empty(8, dtype=np.float64)
                idx = 0
                for dx in range(2):
                    ii = (i + dx) % nk
                    for dy in range(2):
                        jj = (j + dy) % nk
                        for dz in range(2):
                            kk = (k + dz) % nk
                            a_corner[idx] = e_k_grid[ii, jj, kk]
                            b_corner[idx] = e_kq_grid[ii, jj, kk]
                            idx += 1
                # corner flat-index convention: idx = dx*4 + dy*2 + dz

                for t in range(n_tet):
                    a0 = a_corner[tet_offsets[t, 0, 0]*4 + tet_offsets[t, 0, 1]*2 + tet_offsets[t, 0, 2]]
                    a1 = a_corner[tet_offsets[t, 1, 0]*4 + tet_offsets[t, 1, 1]*2 + tet_offsets[t, 1, 2]]
                    a2 = a_corner[tet_offsets[t, 2, 0]*4 + tet_offsets[t, 2, 1]*2 + tet_offsets[t, 2, 2]]
                    a3 = a_corner[tet_offsets[t, 3, 0]*4 + tet_offsets[t, 3, 1]*2 + tet_offsets[t, 3, 2]]
                    b0 = b_corner[tet_offsets[t, 0, 0]*4 + tet_offsets[t, 0, 1]*2 + tet_offsets[t, 0, 2]]
                    b1 = b_corner[tet_offsets[t, 1, 0]*4 + tet_offsets[t, 1, 1]*2 + tet_offsets[t, 1, 2]]
                    b2 = b_corner[tet_offsets[t, 2, 0]*4 + tet_offsets[t, 2, 1]*2 + tet_offsets[t, 2, 2]]
                    b3 = b_corner[tet_offsets[t, 3, 0]*4 + tet_offsets[t, 3, 1]*2 + tet_offsets[t, 3, 2]]

                    for m in range(n_nodes):
                        l0 = lam[m, 0]; l1 = lam[m, 1]; l2 = lam[m, 2]; l3 = lam[m, 3]
                        a = l0*a0 + l1*a1 + l2*a2 + l3*a3
                        b = l0*b0 + l1*b1 + l2*b2 + l3*b3
                        de = a - b
                        xk = beta * (a - mu)
                        nFk = _Fermi(xk)
                        if abs(de) < 1e-12:
                            val = beta * nFk * (1. - nFk)
                        else:
                            xkq = beta * (b - mu)
                            nFkq = _Fermi(xkq)
                            val = -(nFk - nFkq) / de
                        s += qw[m] * val
        partial[i] = s

    total = 0.0
    for i in range(nk):
        total += partial[i]
    return total / Nk


@njit(parallel=True, cache=True)
def matsubara_tetra_3d(mu, beta, e_k_grid, e_kq_grid, S_iw, tet_offsets, lam, qw):
    """
    Tetrahedron-method Matsubara-summed chi0, MOMENTUM-INDEPENDENT self-energy
    only (matches your onsite-only SCBA simplification -- i.e. the
    idx_k = idx_kq = 0 branch of matsubara_ksp). Same normalization convention.

    S_iw : (niw,) complex128   -- Sigma(iw_n), n = 0..niw-1 (fermionic, w_n>0 half)

    COST WARNING: cost scales as nk^3 * 6 * n_nodes * niw. Cheap and clearly
    worth it for niw==1 (static/Kohn-anomaly searches). For large niw, either
    keep n_quad small (3-4) or restrict to a handful of q-points near a
    suspected anomaly rather than a full production sweep -- don't reach for
    this as a full replacement for matsubara_rsp's FFT sweep.

    If your self-energy becomes momentum-dependent again, S would need its
    own linear interpolation across each tetrahedron just like e_k/e_kq --
    that's a straightforward extension of this kernel but isn't implemented
    here since your current SCBA is onsite-only.
    """
    nk = e_k_grid.shape[0]
    Nk = nk * nk * nk
    niw = S_iw.shape[0]
    n_nodes = lam.shape[0]
    n_tet = tet_offsets.shape[0]

    iw_arr = np.empty(niw, dtype=np.complex128)
    for n in range(niw):
        iw_arr[n] = 1j * (2*n + 1) * np.pi / beta

    partial = np.zeros(nk, dtype=np.float64)

    for i in prange(nk):
        s = 0.0
        for j in range(nk):
            for k in range(nk):
                a_corner = np.empty(8, dtype=np.float64)
                b_corner = np.empty(8, dtype=np.float64)
                idx = 0
                for dx in range(2):
                    ii = (i + dx) % nk
                    for dy in range(2):
                        jj = (j + dy) % nk
                        for dz in range(2):
                            kk = (k + dz) % nk
                            a_corner[idx] = e_k_grid[ii, jj, kk]
                            b_corner[idx] = e_kq_grid[ii, jj, kk]
                            idx += 1

                for t in range(n_tet):
                    a0 = a_corner[tet_offsets[t, 0, 0]*4 + tet_offsets[t, 0, 1]*2 + tet_offsets[t, 0, 2]]
                    a1 = a_corner[tet_offsets[t, 1, 0]*4 + tet_offsets[t, 1, 1]*2 + tet_offsets[t, 1, 2]]
                    a2 = a_corner[tet_offsets[t, 2, 0]*4 + tet_offsets[t, 2, 1]*2 + tet_offsets[t, 2, 2]]
                    a3 = a_corner[tet_offsets[t, 3, 0]*4 + tet_offsets[t, 3, 1]*2 + tet_offsets[t, 3, 2]]
                    b0 = b_corner[tet_offsets[t, 0, 0]*4 + tet_offsets[t, 0, 1]*2 + tet_offsets[t, 0, 2]]
                    b1 = b_corner[tet_offsets[t, 1, 0]*4 + tet_offsets[t, 1, 1]*2 + tet_offsets[t, 1, 2]]
                    b2 = b_corner[tet_offsets[t, 2, 0]*4 + tet_offsets[t, 2, 1]*2 + tet_offsets[t, 2, 2]]
                    b3 = b_corner[tet_offsets[t, 3, 0]*4 + tet_offsets[t, 3, 1]*2 + tet_offsets[t, 3, 2]]

                    for m in range(n_nodes):
                        l0 = lam[m, 0]; l1 = lam[m, 1]; l2 = lam[m, 2]; l3 = lam[m, 3]
                        a = l0*a0 + l1*a1 + l2*a2 + l3*a3
                        b = l0*b0 + l1*b1 + l2*b2 + l3*b3

                        acc = 0.0
                        for n in range(niw):
                            Gk = 1.0 / (iw_arr[n] + mu - a - S_iw[n])
                            Gkq = 1.0 / (iw_arr[n] + mu - b - S_iw[n])
                            acc += (Gk * Gkq).real
                        s += qw[m] * acc
        partial[i] = s

    total = 0.0
    for i in range(nk):
        total += partial[i]
    return -2.0 * total / (beta * Nk)


# --------------------------------------------------------------------------
# Example integration with your existing lattice.py / obs.py calling pattern
# --------------------------------------------------------------------------
"""
tets = kuhn_tetrahedra_3d()
lam, qw = build_tetra_quadrature(n_quad=6)   # build once, reuse across all (n, beta, q)

# per q-point, reusing your existing infrastructure:
e_k_full = bz_fine.unfold_f_k(bz_fine.e_k) if bz_fine.ibz else bz_fine.e_k
e_kq_full = get_e_kq(e_k_full, q, nk, method='fft', R_vecs=lat.R_vecs, t_vals=lat.t_vals)

e_k_grid  = e_k_full.reshape((nk, nk, nk))
e_kq_grid = e_kq_full.reshape((nk, nk, nk))

chi0 = lindhard_tetra_3d(mu, beta, e_k_grid, e_kq_grid, tets, lam, qw)
"""