import numpy as np
from triqs.gf import GfImFreq, GfReFreq, MeshReFreq
from scipy.optimize import brentq, minimize, curve_fit
from scripts.utils import *
from scripts.lattice import *
import time
from numba import njit
import sys
from scipy.special import zeta

def get_A_iw0k(G_iwk, n_pade=60):

    niw = G_iwk.data.shape[0]
    N_k  = G_iwk.data.shape[1]

    beta = G_iwk.mesh.components[0].beta
    niw = niw // 2

    G_iw = GfImFreq(beta=beta, statistic='Fermion',
                    n_points=niw, target_shape=[1, 1])

    w_mesh = MeshReFreq(w_min=0.0, w_max=0.0, n_w=1)
    G_w = GfReFreq(mesh=w_mesh, target_shape=[1, 1])

    n_pade = min(n_pade, niw)

    A_k = np.empty(N_k)

    for ik in range(N_k):
        G_iw.data[:, 0, 0] = G_iwk.data[:, ik, 0, 0]

        G_w.set_from_pade(G_iw, n_points=n_pade)

        A_k[ik] = -G_w.data[0, 0, 0].imag / np.pi

    return A_k

def get_Z(S_iw_data, beta, niw):
    """
    Evaluate quasi-particle weight Z, scattering rate gamma, and lifetime tau from self-energy.
    """
    
    n = np.arange(niw)
    iw = (2*n + 1) * np.pi / beta

    iw0 = iw[0]
    S0 = S_iw_data[niw].imag

    Z = 1. / (1. - S0 / iw0)
    gamma = -Z * S0
    tau = 1. / gamma

    return Z, gamma, tau

@njit
def cexp(z):
    zr = z.real
    zi = z.imag
    ezr = np.exp(zr)
    return ezr * np.cos(zi) + 1j * ezr * np.sin(zi)

@njit
def Fermi(z_k):
    z_k_real = z_k.real
    if z_k_real > 500.:
        nF_k = 0. + 0.j
    elif z_k_real < -500.:
        nF_k = 1. + 0.j
    else:
        nF_k = 1. / (cexp(z_k) + 1.)

    return nF_k

@njit
def dFermi(z_k):
    z_k_real = z_k.real
    if z_k_real > 500. or z_k_real > 500.:
        dnF_k = 0. + 0.j
    else:
        dnF_k = -cexp(z_k) / (cexp(z_k) + 1.)**2
    return dnF_k

@njit
def lindhard_ksp(mu, beta, e_k, e_kq, S_k, S_kq, de_kq_dq=None):
    
    Nk = len(e_k)
    if de_kq_dq is not None:
        dim = de_kq_dq.shape[0]
        dchi0_sum = np.zeros(dim)

    lenS_k = len(S_k)
    lenS_kq = len(S_kq)

    chi0_sum = 0.j
    for k in range(Nk):
        s_k = S_k[min(lenS_k - 1, k)]
        s_kq = S_kq[min(lenS_kq - 1, k)]

        e_k_eff = e_k[k] + s_k.real
        e_kq_eff = e_kq[k] + s_kq.real
        Gamma_k = -s_k.imag
        Gamma_kq = -s_kq.imag

        xk = beta * (e_k_eff + 1j*Gamma_k - mu)
        xkq = beta * (e_kq_eff + 1j*Gamma_kq - mu)

        nF_k = Fermi(xk)
        nF_kq = Fermi(xkq)

        de = (e_k_eff - e_kq_eff) - 1j*(Gamma_k - Gamma_kq)

        if abs(de) < 1e-8:
            chi0_k = beta * nF_k * (1.0 - nF_k)
        else:
            chi0_k = -(nF_k - nF_kq) / de
        
        chi0_sum += chi0_k
        
        if de_kq_dq is not None and abs(de) > 1e-8:
            dnF_kq = dFermi(xkq)
            for alpha in range(dim):
                v = de_kq_dq[alpha, k]
                num = (dnF_kq * beta * v) * de - (nF_k - nF_kq) * v
                dchi0_sum[alpha] += (num / de**2).real

    chi0 = chi0_sum.real / Nk

    if de_kq_dq is None:
        dchi0 = None

    else: 
        dchi0 = np.zeros(dim)
        for alpha in range(dim):
            dchi0[alpha] = dchi0_sum[alpha] / Nk
    
    return chi0, dchi0

@njit
def matsubara_ksp(mu, beta, e_k, e_kq, S_iwk, S_iwkq):
    
    Nk = len(e_k)
    niw = len(S_iwk)

    chi0 = 0.0j
    for n in range(niw):
        iw = 1j * (2*n + 1) * np.pi / beta
        for k in range(Nk):
            G_iwk  = 1.0 / (iw + mu - e_k[k] - S_iwk[n,k])
            G_iwkq = 1.0 / (iw + mu - e_kq[k] - S_iwkq[n,k])
            chi0 += G_iwk * G_iwkq
    
    tail = beta/(2*np.pi**2) * zeta(2, niw + 0.5)

    return (-2.0 / beta * chi0).real / Nk + tail

def matsubara_rsp(lat, beta, G_iwk, nk, ibz):

    dim = lat.dim
    niw = G_iwk.shape[0]

    G_r = np.empty((nk,)*dim, dtype=np.complex128)
    chi0_r = np.zeros((nk,)*dim, dtype=np.float64)
    neg = np.r_[0, np.arange(nk-1, 0, -1)]

    for n in range(niw):
        if ibz:
            G_k = lat.unfold_f_k(G_iwk[n]).reshape((nk,)*dim)
        else:
            G_k = G_iwk[n].reshape((nk,)*dim)
        np.fft.ifftn(G_k, out=G_r)
        chi0_r += (G_r * G_r[neg]).real

    chi0_q  = np.fft.fftn(chi0_r)
    chi0_q *= -2.0 / beta

    tail = beta/(2*np.pi**2) * zeta(2, niw + 0.5)
    chi0_q += tail

    invchi0_q = 1.0 / chi0_q.real
    return invchi0_q.reshape(-1)

@njit
def get_G_iwk(mu, beta, e_k, S_iwk, niw):
    
    Nk = len(e_k)

    len_S_k = len(S_iwk[0])
    len_S_iw = len(S_iwk[1])

    G_iwk = np.zeros((niw, Nk), dtype=np.complex128)
    for n in range(niw):
        iw_n = 1.j * (2*n + 1) * np.pi / beta
        for k in range(Nk):
            s_iwk = S_iwk[min(len_S_iw - 1, n), min(len_S_k - 1, k)]
            G_iwk[n,k] = 1. / (iw_n + mu - e_k[k] - s_iwk)
    
    return G_iwk        

# Does not yet handle (niw, Nk_fine) or (Nk_fine,) as shapes, relevant only for k_dep
def get_iwk_arr(S_val, Nk, niw=1):
    
    k_dep = False

    # If S_val is None or a scalar, return (niw, 1) array
    if S_val is None or isinstance(S_val, (int, float, complex)):
        val = 0j if S_val is None else complex(S_val)
        S_val = np.full((niw, 1), val, dtype=complex)

    else:
        S_val = np.asanyarray(S_val, dtype=complex)

        # If S_val is either (niw,) or (Nk,)
        if S_val.ndim == 1:
            if len(S_val) == Nk:
                S_val = np.tile(S_val[np.newaxis, :], (niw, 1))  # return (niw, Nk)
                k_dep = True
            elif len(S_val) == niw:
                S_val = S_val[:, np.newaxis]  # return (niw, 1)
        elif S_val.ndim == 2:
            if S_val.shape == (niw, Nk):
                k_dep = True
        else:
            print(np.shape(S_val))
            print(niw, Nk)
            print('S_val must be either None, scalar or array/list with shape (niw,), (Nk,) or (niw, Nk)')
            sys.exit()

    return S_val, k_dep

def get_grid_from_path(q_path, k_grid, tol=None):

    start = np.asarray(q_path[0], dtype=float)
    stop  = np.asarray(q_path[-1], dtype=float)

    q_grid_full = np.asarray(k_grid, dtype=float) / np.pi

    direction = stop - start
    length = np.linalg.norm(direction)

    unit_dir = direction / length

    # vectors from start to all grid points
    diff = q_grid_full - start

    # coordinate along the path
    t = diff @ unit_dir

    # perpendicular distance from the line
    perp = diff - np.outer(t, unit_dir)
    dist = np.linalg.norm(perp, axis=1)

    # automatic tolerance from nearest-neighbor spacing
    if tol is None:
        spacing = np.min(np.linalg.norm(
            q_grid_full[1:] - q_grid_full[:-1], axis=1
        ))
        tol = 0.51 * spacing

    path_mask = (
        (dist < tol) &
        (t >= -tol) &
        (t <= length + tol)
    )

    q_grid = q_grid_full[path_mask]
    t_sel = t[path_mask]

    # sort along the path
    order = np.argsort(t_sel)
    q_grid = q_grid[order]

    # explicitly include endpoints
    #if np.linalg.norm(q_grid[0] - start) > tol:
    #    q_grid = np.vstack([start, q_grid])

    #if np.linalg.norm(q_grid[-1] - stop) > tol:
    #    q_grid = np.vstack([q_grid, stop])

    return q_grid, path_mask

def get_invchi0_min(lat, mu, beta, q_path=None, method='matsubara', niw=1, S_val=None, ibz=True):
    
    e_k = lat.e_k
    Nk = len(e_k)

    # q_grid either from q_path or from the BZ/IBZ
    if q_path is not None:
        q_grid, _ = get_grid_from_path(q_path, lat.k_vecs)
    else:
        q_grid = lat.k_vecs / np.pi

    # From S_val to (niw, Nk) array
    S_iwk, k_dep = get_iwk_arr(S_val, Nk, niw)

    if method == 'lindhard':

        # Unfolding e_k and S_iwk into the full BZ
        if ibz:
            e_k = lat.unfold_f_k(e_k)
            if k_dep:
                S_iwk = lat.unfold_f_iwk(S_iwk)
        
        # 1/chi0 for each q using Lindhard
        invchi0_grid = []
        for q in q_grid:
            e_kq = lat.get_e_kq(e_k, q, lat.nk)

            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, lat.nk)
            else:
                S_iwkq = S_iwk

            chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_iwkq[0])
            invchi0_grid.append(1/chi0)

    elif method == 'matsubara':

        G_iwk = np.array(get_G_iwk(mu, beta, e_k, S_iwk, niw))
        
        # Evaluate 1/chi0 over whole BZ using FFT back and forth
        invchi0_grid = matsubara_rsp(lat, beta, G_iwk, lat.nk, ibz)

        # Fold into IBZ to save memory
        if ibz:
            invchi0_grid = lat.fold_f_k(invchi0_grid)

    # Minimum of 1/chi0 over q_grid
    idx = np.argmin(invchi0_grid)
    q_min = q_grid[idx]
    invchi0_min = invchi0_grid[idx]

    return np.array(q_min), invchi0_min, np.array(invchi0_grid)

def search_min(lat, mu, beta, q_min, q_path=None, S_val=None, refine_ratio=1, ibz=True):
    
    dim = lat.dim
    e_k = lat.e_k_fine
    Nk = len(e_k)

    # q_grid either from q_path or from the BZ/IBZ
    if q_path is not None:
        q_grid, _ = get_grid_from_path(q_path, lat.k_vecs)
    else:
        q_grid = lat.k_vecs / np.pi

    # From S_val to (1, Nk) array
    S_iwk, k_dep = get_iwk_arr(S_val, Nk, 1)

    # Get new nk, e_k and S_iwk
    e_k = lat.e_k_fine
    if ibz:
        e_k = lat.unfold_f_k(e_k, fine=True)

    if k_dep:
        if ibz:
            S_iwk = lat.unfold_f_iwk(S_iwk, fine=True)

        # If k_dep, precalculate S_iwR for the FFT
        S_iwR = lat.get_f_iwR(S_iwk, lat.nk_fine, fine=True)

    # Evaluates e_kq/S_iwkq shifting the arrays by q with np.roll
    def invchi0_q_roll(s):
        q = start + np.dot(J, np.atleast_1d(s))

        e_kq = lat.get_e_kq(e_k, q, lat.nk_fine)
        if k_dep:
            S_iwkq = lat.get_f_iwkq(S_iwk, q, lat.nk_fine)
        else:
            S_iwkq = S_iwk

        chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_iwkq[0])

        return 1/chi0

    # Evaluates e_kq/S_iwkq with the exact formulas, requires t_vals and S_iwR
    def invchi0_q_exact(s):
        q = start + np.dot(J, np.atleast_1d(s))

        e_kq, de_kq_dq = lat.get_e_kq(e_k, q, lat.nk_fine, method='exact')
        if k_dep:
            S_iwkq = lat.get_f_iwkq(S_iwk, q, lat.nk_fine, method='exact', f_iwR=S_iwR)
        else:
            S_iwkq = S_iwk

        chi0, dchi0_dq = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_iwkq[0], de_kq_dq=de_kq_dq)

        dinvchi0_dq = -dchi0_dq / chi0**2
        grad_s = J.T @ dinvchi0_dq

        return 1/chi0, grad_s
    
    # If needed you can increase the borders of the refinement region
    safe = 1

    if q_path is not None:
        start = q_grid[0]
        stop = q_grid[-1]
        J     = (stop - start).reshape(dim, 1)
        J_inv = np.linalg.pinv(J)
        step  = 1 / (len(q_grid) - 1)
        s0 = J_inv @ (q_min - start)
        lo    = max(0.0, s0[0] - safe*step)
        hi    = min(1.0, s0[0] + safe*step)
        bounds = [(lo, hi)]

    else:
        start = q_grid[0]
        J = lat.b_vecs / (2*np.pi)
        J_inv = np.linalg.inv(J)
        s0 = J_inv @ (q_min - start)
        step = 1.0 / lat.nk
        bounds = []
        for d in range(dim):
            lo = max(0.0, s0[d] - safe*step)
            hi = min(1.0, s0[d] + safe*step)
            bounds.append((lo, hi))

    if refine_ratio > 1.:

        if q_path is not None:
            q_grid_fine, _ = get_grid_from_path(q_path, lat.k_vecs_fine)
            start = q_grid_fine[0]
            s_grid_fine = np.array([J_inv @ (q - start) for q in q_grid_fine]).squeeze()
            mask = (s_grid_fine >= lo) & (s_grid_fine <= hi)
            s_grid = s_grid_fine[mask]

        else:
            q_grid_fine = lat.k_vecs_fine / np.pi
            s_grid_fine = np.array([J_inv @ (q - start) for q in q_grid_fine])
            mask = np.all((s_grid_fine >= [b[0] for b in bounds]) & 
                        (s_grid_fine <= [b[1] for b in bounds]), axis=1)
            s_grid = s_grid_fine[mask]
        
        invchi0_grid_fine = np.array([invchi0_q_roll(s) for s in s_grid])
        s0 = np.atleast_1d(s_grid[np.argmin(invchi0_grid_fine)])
        
        if q_path is not None:
            step  = 1 / (len(q_grid_fine) - 1)
            lo    = max(0.0, s0[0] - safe*step)
            hi    = min(1.0, s0[0] + safe*step)
            bounds = [(lo, hi)]
        else:
            step  = 1.0 / lat.nk_fine
            bounds = []
            for d in range(dim):
                lo = max(0.0, s0[d] - safe*step)
                hi = min(1.0, s0[d] + safe*step)
                bounds.append((lo, hi))

    res = minimize(
        invchi0_q_exact,
        x0=s0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'ftol': 1e-6, 'gtol': 1e-4, 'maxiter': 50}
    )
    s_min = res.x

    q_min = start + J @ s_min
    invchi0_min, _ = invchi0_q_exact(s_min)

    return np.array(q_min), invchi0_min

def fit_invxi(lat, mu, beta, q_min, U=None, q_path=None, invchi_grid=None, S_val=None, fit_range=[0,0,1e-2], fit_pts=15, ibz=True, fit_grid_pts=True, always_fit_qmin=False):

    if lat.nk_fine is None:
        nk = lat.nk
        e_k = lat.e_k
        fine = False
    else:
        nk = lat.nk_fine
        e_k = lat.e_k_fine
        fine = True

    Nk = len(e_k)
    S_iwk, k_dep = get_iwk_arr(S_val, Nk, 1)

    # q_grid either from q_path or from the BZ/IBZ
    if q_path is not None:
        q_grid, _ = get_grid_from_path(q_path, lat.k_vecs)
    else:
        q_grid = lat.k_vecs / np.pi
    
    start = q_min - fit_range
    stop = q_min + fit_range
    
    if not fit_grid_pts:
        
        q_grid = np.linspace(start, stop, fit_pts)

        if ibz:
            e_k = lat.unfold_f_k(e_k, fine=fine)
            if k_dep:
                S_iwk = lat.unfold_f_iwk(S_iwk, fine=fine)
        
        # 1/chi0 for each q using Lindhard
        chi_grid = []
        for q in q_grid:
            e_kq, _ = lat.get_e_kq(e_k, q, nk, method='exact')

            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, nk, method='exact')
            else:
                S_iwkq = S_iwk

            chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_iwkq[0])
            chi_grid.append(chi0/(1 - U*chi0))
    
    else:
        q_path = (start, stop)
        q_grid, path_mask = get_grid_from_path(q_path, np.pi*q_grid)
        chi_grid = 1/np.array(invchi_grid)[path_mask]
    
    p0 = [0.01, 0., abs(1/chi_grid[len(chi_grid)//2])]
    bounds = ([0., -np.inf, 0.], [1., np.inf, np.inf])

    e_hat = (stop - start)
    e_hat /= np.linalg.norm(e_hat)
    s_grid = q_grid @ e_hat
    s0 = q_min @ e_hat

    if not fine or always_fit_qmin:
        p0.append(s0)
        bounds[0].append(s0 - 4/nk)
        bounds[1].append(s0 + 4/nk)
        OZ_fit = OZ
    else:
        OZ_fit = lambda a, b, invxi: OZ(a, b, invxi, s0)

    try:
        par, _ = curve_fit(OZ_fit, s_grid, chi_grid, p0=p0, bounds=bounds, maxfev=10000)
        if not fine or always_fit_qmin:
            new_q_min = q_min - (q_min @ e_hat)*e_hat + par[3]*e_hat
        else:
            new_q_min = np.array([np.nan]*len(q_min))
    
    except RuntimeError:
        n_pars = len(p0)
        par = np.array([np.nan]*n_pars)
        new_q_min = np.array([np.nan]*len(q_min))

    return par, new_q_min

def density_k(e_k, S_k, mu, beta, ibz_w_k):
    x  = e_k + S_k.real - mu
    xr = np.clip(beta * x, -500, 500)
    xi = beta * S_k.imag

    cos_xi = np.cos(xi)

    ex  = np.exp(xr)
    emx = np.exp(-xr)

    nF_real = (cos_xi + emx) / (ex + 2*cos_xi + emx)

    return 2. * np.dot(nF_real, ibz_w_k)/np.sum(ibz_w_k)

@njit
def density_iwk(e_k, Gamma_k, mu, beta, ibz_w_k):
    Nk = e_k.shape[0]
    niw = Gamma_k.shape[0]
    n_tot = 0.0
    for k in range(Nk):
        xi = e_k[k] - mu
        Gamma = Gamma_k[0, 0]
        s = 0.0 + 0.0j
        for n in range(niw):
            wn = (2*n + 1) * np.pi / beta
            z = 1j*wn - xi + 1j*Gamma
            s += 1.0 / z - 1.0 / (1j*wn) - xi / wn**2  # subtract tails
        nk = 0.5 + xi/2 + 2*(s / beta).real             # add back analytic tails
        n_tot += ibz_w_k[k] * nk
    return 2*n_tot/sum(ibz_w_k)

def get_mu(e_k, n_goal, beta, niw=1, S_iwk=None, ibz_w_k=None):
    
    Nk_ibz = len(e_k)
    S_iwk, _ = get_iwk_arr(S_iwk, Nk=Nk_ibz, niw=niw)

    if ibz_w_k is None:
        ibz_w_k = np.ones(Nk_ibz)

    if niw > 1:
        def density(mu): return density_iwk(e_k, S_iwk, mu, beta, ibz_w_k)
    else:
        def density(mu): return density_k(e_k, S_iwk[0], mu, beta, ibz_w_k)

    def f(mu): return density(mu) - n_goal

    a = e_k.min() - 50.0/beta
    b = e_k.max() + 50.0/beta
    
    return brentq(f, a, b)

def initialize_bz(lat, nk, refine=True, refine_ratio=1, ibz=True):

    if lat.nk is None:
        lat.get_bz(nk, ibz=ibz)
        lat.get_e_k()

    if lat.nk_fine is None and refine:
        nk_fine = int(refine_ratio*nk)
        lat.get_bz(nk_fine, ibz=ibz, fine=True)
        lat.get_e_k(fine=True)

def run_rpa(par, lat=None, t=1., tp=0., dim=3, nk=100, niw=1, S_val=None, q_path=None, method='matsubara', refine=True,
            refine_ratio=1, get_xi=False, xi_range=[0,0,1e-2], xi_pts=15, fit_grid_pts=True, always_fit_qmin=False,  ibz=True, save_invchigrid=False, verbose=True, save_inputs=True):
    
    if verbose:
        print('='*50)
        start_time = time.time()

    # Stores every input inside a dictionary
    if save_inputs:
        run_data = locals().copy()
    else:
        run_data = {}

    if lat is None:
        lat = LATTICE(t=t, tp=tp, dim=dim)
        initialize_bz(lat, nk, refine, refine_ratio, ibz)
    else:
        refine = lat.nk_fine is not None
        refine_ratio = lat.nk_fine/lat.nk if refine else 1
        ibz = lat.ibz

    U, T, n = par['U'], par['T'], par['n']

    if verbose: print(f'U={U:.3f}, T={T:.3f}, n={n:.3f}: self-consistent mu search...', end='')

    finer_e_k = lat.e_k_fine if lat.e_k_fine is not None else lat.e_k
    finer_ibz_w_k = lat.ibz_w_k_fine if lat.ibz_w_k_fine is not None else lat.ibz_w_k
    mu = get_mu(e_k=finer_e_k, n_goal=n, beta=1/T, S_iwk=S_val, ibz_w_k=finer_ibz_w_k)

    if verbose: print(f'U={U:.3f}, T={T:.3f}, n={n:.3f}: 1/chi0 minimum search over grid with nk={lat.nk}...', end='')
    Q, invchi0_Q, invchi0_grid = get_invchi0_min(lat, mu, 1/T, q_path, method, niw, S_val, ibz)
    
    if refine and niw==1:
        if verbose: print(f'U={U:.3f}, T={T:.3f}, n={n:.3f}: 1/chi0 minimum refinement with nk\'={lat.nk_fine}...', end='')
        Q, invchi0_Q = search_min(lat, mu, 1/T, Q, q_path, S_val, refine_ratio, ibz)
    
    invchi_Q = invchi0_Q.real - U
    if save_invchigrid:
        run_data['invchi_grid'] = invchi0_grid.real - U
    
    if get_xi:
        if invchi_Q > 0.:
            if verbose: print(f'U={U:.3f}, T={T:.3f}, n={n:.3f}: 1/xi estimation with OZ fit over chi(q)...', end='')
            run_data['OZ_fit'], run_data['Q_fitted'] = fit_invxi(lat, mu, 1/T, Q, U, q_path, invchi0_grid.real-U, S_val, xi_range, xi_pts, ibz, fit_grid_pts, always_fit_qmin)
            invxi = run_data['OZ_fit'][2]
            if not refine and niw > 1:
                Q = run_data['Q_fitted']
        else:
            n_pars = 4 if not refine else 3
            run_data['OZ_fit'] = np.array([np.nan]*n_pars)
            invxi = np.nan
            run_data['Q_fitted'] = np.array([np.nan]*dim)

    if verbose:
        Q_str = "(" + ", ".join(f"{x:.3g}" for x in Q) + ")"
        invxi_str = "None" if invxi is None else f"{invxi:.3f}"
        elapsed_time = time.time() - start_time
        print(f"U={U:.3f}, T={T:.3f}, n={n:.3f}: completed after {elapsed_time:.1f}s"
              f"results: 1/chi0 = {invchi_Q:.3f}, Q = {Q_str}, 1/xi = {invxi_str}", end='')

    run_data['invchi'] = invchi_Q
    run_data['Q'] = Q
    run_data['mu'] = mu
    if get_xi:
        run_data['invxi'] = invxi
    else:
        run_data['invxi'] = np.nan
    
    return run_data
          
def sweep_rpa(par_list, lat=None, t=1., tp=0., dim=3, nk=100, niw=1, S_list=None, q_path=None, method='matsubara', 
                 refine=True, refine_ratio=1., get_xi=False, xi_range=[0,0,1e-2], xi_pts=15, fit_grid_pts=True, always_fit_qmin=False, fit=False, fit_type=HMM, ibz=True, save_file=None, verbose=True, workers=6):

    if verbose:
        print('='*50)
        start_time = time.time()

    # Extract the parameter list for the loop
    sweep_length = len(par_list)
    varying_key = []
    for key in par_list[0].keys():
        values = {par[key] for par in par_list}
        if len(values) > 1:
            varying_key.append(key)
    
    if len(varying_key) > 1:
        print("Only one parameter can vary among U, T and n for each sweep!")
        sys.exit()
    else:
        varying_key = varying_key[0]
    
    varying_par = [par[varying_key] for par in par_list]

    if isinstance(S_list, (list, np.ndarray)) and len(S_list) == sweep_length:
        pass
    else:
        S_list = [S_list] * sweep_length

    if lat is None:
        lat = LATTICE(t=t, tp=tp, dim=dim)
        initialize_bz(lat, nk, refine, refine_ratio, ibz)
    else:
        refine = lat.nk_fine is not None
        refine_ratio = lat.nk_fine/lat.nk if refine else 1
        ibz = lat.ibz

    inputs = [{'lat': lat, 't': t, 'tp': tp, 'dim': dim, 'nk': lat.nk, 'niw': niw, 'S_val': S_list[i], 'q_path': q_path,'method': method, 'refine': refine, 'refine_ratio': refine_ratio, 'get_xi': get_xi, 'xi_range': xi_range, 'xi_pts': xi_pts, 'fit_grid_pts': fit_grid_pts, 'always_fit_qmin': always_fit_qmin, 'ibz': ibz, 'verbose': False, 'save_inputs': False, 'par': par_list[i]} for i in range(sweep_length)]
    
    results_list = run_parallel(run_rpa, inputs, workers=workers)
    
    results = {key: np.array([d[key] for d in results_list]) for key in results_list[0]}

    if fit:
        if not isinstance(fit_type, (list, np.ndarray)):
            fit_type = [fit_type]
        
        results['fitchi'] = {}
        if get_xi:
            results['fitxi'] = {}

        par_keys = ['a', 'b', 'Xc']
        for label in par_keys:
            results['fitchi'][label] = []
            results['fitchi'][f'{label}_err'] = []

            if get_xi:
                results['fitxi'][label] = []
                results['fitxi'][f'{label}_err'] = []

        results['Qc'], results['mu_c'] = [], []

        for fit in fit_type:

            labels_to_fit = ['chi']
            if get_xi:
                labels_to_fit.append('xi')

            for label in labels_to_fit:
                pos_mask = results[f'inv{label}'] > 0
                x_fit = np.array(varying_par)[pos_mask][:15]
                y_fit = results[f'inv{label}'][pos_mask][:15]

                if len(x_fit) >= 2:
                    try:
                        p0 = [1., 1., np.min(x_fit) * 0.9]
                        bounds = ([0., 0., -np.inf], [np.inf, np.inf, np.inf])
                        par, cov = curve_fit(fit, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)

                        for i, key in enumerate(par_keys):
                            results[f'fit{label}'][key].append(par[i])
                            err = np.sqrt(cov[i, i])
                            results[f'fit{label}'][f'{key}_err'].append(err)
                        
                        if label == 'chi':
                            Q_vals = results['Q'][pos_mask][:2]
                            m = (Q_vals[1] - Q_vals[0])/(x_fit[1] - x_fit[0])
                            results['Qc'].append(Q_vals[0] - m * (x_fit[0] - results[f'fit{label}']['Xc']))

                            mu_vals = results['mu'][pos_mask][:2]
                            m = (mu_vals[1] - mu_vals[0])/(x_fit[1] - x_fit[0])
                            results['mu_c'].append(mu_vals[0] - m * (x_fit[0] - results[f'fit{label}']['Xc']))
                        
                        converged = True

                    except RuntimeError:
                        print(f"Fit of inv{label} values did not converge")
                        converged = False
                else:
                    print("Not enough points to fit!")
                    converged = False

                if not converged:
                    for i, key in enumerate(par_keys):
                        results[f'fit{label}'][key].append(np.nan)
                        results[f'fit{label}'][f'{key}_err'].append(np.nan)

                    if label == 'chi':
                        results['Qc'].append(np.array([np.nan]*dim))
                        results['mu_c'].append(np.nan)
                
        if len(fit_type) == 1:
            for label in labels_to_fit:
                for key in par_keys:
                    results[f'fit{label}'][key] = results[f'fit{label}'][key][0]
                    results[f'fit{label}'][f'{key}_err'] = results[f'fit{label}'][f'{key}_err'][0]

            results['Qc'] = results['Qc'][0]
            results['mu_c'] = results['mu_c'][0]

    sweep_data = locals().copy()
    sweep_data.update(results)

    if save_file is not None:
        HDFwrite_dict(save_file, sweep_data)
    
    const_keys = list(par_list[0].keys() - {varying_key})
    const_pars = [par_list[0][key] for key in const_keys]
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\r{const_keys[0]}: {const_pars[0]:.3f}, {const_keys[1]}: {const_pars[1]:.3f}: "
              f"Completed {sweep_length} jobs in {elapsed_time:.1f} seconds")

    return sweep_data