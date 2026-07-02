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
    if z_k > 500.:
        nF_k = 0.
    elif z_k < -500.:
        nF_k = 1.
    else:
        nF_k = 1. / (np.exp(z_k) + 1.)

    return nF_k

@njit
def dFermi(z_k):
    if z_k > 500. or z_k < -500.:
        dnF_k = 0.
    else:
        dnF_k = -np.exp(z_k) / (np.exp(z_k) + 1.)**2
    return dnF_k

@njit
def lindhard_ksp(mu, beta, e_k, e_kq, de_kq_dq=None):
    
    Nk = len(e_k)
    if de_kq_dq is not None:
        dim = de_kq_dq.shape[0]
        dchi0 = np.zeros(dim)

    chi0 = 0.
    for k in range(Nk):

        e_k_eff = e_k[k]
        e_kq_eff = e_kq[k]

        xk = beta * (e_k_eff - mu)
        xkq = beta * (e_kq_eff - mu)

        nF_k = Fermi(xk)
        nF_kq = Fermi(xkq)

        de = e_k_eff - e_kq_eff

        if abs(de) < 1e-8:
            chi0_k = beta * nF_k * (1.0 - nF_k)
        else:
            chi0_k = -(nF_k - nF_kq) / de
        
        chi0 += chi0_k
        
        if de_kq_dq is not None and abs(de) > 1e-8:
            dnF_kq = dFermi(xkq)
            for d in range(dim):
                v = de_kq_dq[d, k]
                num = (dnF_kq * beta * v) * de - (nF_k - nF_kq) * v
                dchi0[d] += num / de**2

    if de_kq_dq is None:
        dchi0 = None
    else:
        dchi0 /= Nk
    
    return chi0/Nk, dchi0

@njit
def matsubara_ksp(mu, beta, e_k, e_kq, S_iwk, S_iwkq):
    
    Nk = len(e_k)
    niw = len(S_iwk)
    lenS_k = len(S_iwk[0])
    lenS_kq = len(S_iwkq[0])

    chi0 = 0.0j
    for n in range(niw):
        iw = 1j * (2*n + 1) * np.pi / beta
        for k in range(Nk):
            G_iwk  = 1.0 / (iw + mu - e_k[k] - S_iwk[n, min(lenS_k - 1, k)])
            G_iwkq = 1.0 / (iw + mu - e_kq[k] - S_iwkq[n, min(lenS_kq - 1, k)])
            chi0 += G_iwk * G_iwkq

    return -2.0 * chi0.real / beta

def matsubara_rsp(bz, beta, G_iwk):

    dim = bz.dim
    nk = bz.nk
    niw = G_iwk.shape[0]

    G_iwk 

    G_r = np.empty((nk,)*dim, dtype=np.complex128)
    chi0_r = np.zeros((nk,)*dim, dtype=np.float64)
    neg = np.r_[0, np.arange(nk-1, 0, -1)]

    for n in range(niw):
        if bz.ibz:
            G_k = bz.unfold_f_k(G_iwk[n]).reshape((nk,)*dim)
        else:
            G_k = G_iwk[n].reshape((nk,)*dim)
        np.fft.ifftn(G_k, out=G_r)
        chi0_r += (G_r * G_r[neg]).real

    chi0_q  = -2.0 / beta * np.fft.fftn(chi0_r).real

    return chi0_q.reshape(-1)

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

@njit
def get_G_iw_slice(mu, beta, e_k, S_iwk, n):
    
    Nk = len(e_k)
    niw = S_iwk.shape[0]
    G = np.empty(Nk, dtype=np.complex128)
    iw_n = 1j * (2*n + 1) * np.pi / beta
    for k in range(Nk):
        sn = min(niw - 1, n)
        sk = min(S_iwk.shape[1] - 1, k)
        G[k] = 1.0 / (iw_n + mu - e_k[k] - S_iwk[sn, sk])
    return G

# Does not yet handle (niw, Nk_fine) or (Nk_fine,) as shapes, relevant only for k_dep
def get_iwk_arr(S_val, Nk, niw=1):
    
    k_dep = False

    # If S_val is None or a scalar, return (niw, 1) array
    if S_val is None or isinstance(S_val, (int, float, complex)):
        val = 0j if S_val is None else complex(S_val)
        S_val = np.full((niw, 1), val, dtype=complex)

    else:
        S_val = np.asarray(S_val, dtype=np.complex128)

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

    return q_grid, path_mask

def niw_extrapolate(invchi0_func, niw, deg=2):

    invchi0_list, niw_list = [], []
    for s in [1, 2, 3, 4]:
        n_sub = niw // s
        if n_sub < 4:
            break

        mesh_sub = np.asarray(invchi0_func(n_sub))

        invchi0_list.append(mesh_sub)
        niw_list.append(n_sub)

    x = 1. / np.asarray(niw_list)
    y = np.stack(invchi0_list, axis=0)

    coeff = np.polyfit(x, y, deg=deg)
    invchi0_mesh = coeff[-1].squeeze()

    return invchi0_mesh

def get_invchi0_grid(bz, mu, beta, q_path=None, method='fft', niw=1, S_val=None, niw_fit=False):
    
    e_k = bz.e_k
    Nk = len(e_k)

    # q_grid either from q_path or from the BZ/IBZ
    if q_path is not None:
        q_grid, _ = get_grid_from_path(q_path, bz.k_vecs)
    else:
        q_grid = bz.k_vecs / np.pi

    # From S_val to (niw, Nk) array
    S_iwk, k_dep = get_iwk_arr(S_val, Nk, niw)

    if method == 'local':

        # Unfolding e_k and S_iwk into the full BZ
        if bz.ibz:
            e_k = bz.unfold_f_k(e_k)
            if k_dep:
                S_iwk = bz.unfold_f_iwk(S_iwk)
        
        # 1/chi0 for each q using Lindhard
        invchi0_grid = []
        for q in q_grid:
            e_kq = get_e_kq(e_k, q, bz.nk)

            if k_dep:
                S_iwkq = get_f_iwkq(S_iwk, q, bz.nk)
            else:
                S_iwkq = S_iwk

            if niw == 1:
                chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq)
            else:
                chi0 = matsubara_ksp(mu, beta, e_k, e_kq, S_iwk, S_iwkq)
                chi0 += beta / (2*np.pi**2) * zeta(2, niw + 0.5)

            invchi0_grid.append(1/chi0)

    elif method == 'fft':

        G_iwk = np.asarray(get_G_iwk(mu, beta, e_k, S_iwk, niw))

        if niw_fit:
            chi0_func = lambda niw: matsubara_rsp(bz, beta, G_iwk[:niw+1])
            chi0_grid = niw_extrapolate(chi0_func, niw)
        else:
            tail = beta / (2*np.pi**2) * zeta(2, niw + 0.5)
            chi0_grid = matsubara_rsp(bz, beta, G_iwk) + tail

        del G_iwk

        # Fold into IBZ to save memory
        if bz.ibz:
            chi0_grid = bz.fold_f_k(chi0_grid)
        
        invchi0_grid = 1/chi0_grid

    # Minimum of 1/chi0 over q_grid
    idx = np.argmin(invchi0_grid)
    q_min = q_grid[idx]
    invchi0_min = invchi0_grid[idx]

    return np.asarray(q_min), invchi0_min, np.asarray(invchi0_grid)

def search_min(lat, bz, bz_fine, mu, beta, q_min, q_path=None):
    
    dim = bz.dim

    # q_grid either from q_path or from the BZ/IBZ
    if q_path is not None:
        q_grid, _ = get_grid_from_path(q_path, bz.k_vecs)
    else:
        q_grid = bz.k_vecs / np.pi

    # Get new nk, e_k and S_iwk
    e_k = bz_fine.e_k
    if bz_fine.ibz:
        e_k = bz_fine.unfold_f_k(e_k)

    # Evaluates e_kq/S_iwkq with the exact formulas, requires t_vals and S_iwR
    def invchi0_q_exact(s):
        q = start + np.dot(J, np.atleast_1d(s))

        e_kq = get_e_kq(e_k, q, bz_fine.nk, method='fft', R_vecs=lat.R_vecs, t_vals=lat.t_vals, k_full=bz_fine.k_full)
        de_kq_dq = get_de_kq_dq(e_k, q, bz_fine.nk, R_vecs=lat.R_vecs, t_vals=lat.t_vals)

        chi0, dchi0_dq = lindhard_ksp(mu, beta, e_k, e_kq, de_kq_dq=de_kq_dq)

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
        step = 2.0 / lat.nk
        bounds = []
        for d in range(dim):
            lo = max(0.0, s0[d] - safe*step)
            hi = min(2.0, s0[d] + safe*step)
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

    return np.asarray(q_min), invchi0_min

def fit_invxi(lat, bz, bz_fine, mu, beta, q_min, U=None, q_path=None, invchi_grid=None, fit_range=[0,0,7e-3], fit_pts=10, fit_grid_pts=True, fit_qmin=False):

    finer_bz = bz_fine if bz_fine is not None else bz

    nk = finer_bz.nk
    e_k = finer_bz.e_k
    
    start = q_min - fit_range
    stop = q_min + fit_range
    
    if not fit_grid_pts:
        
        q_grid = np.linspace(start, stop, fit_pts)

        if finer_bz.ibz:
            e_k = finer_bz.unfold_f_k(e_k)
        
        # 1/chi0 for each q using Lindhard
        chi_grid = []
        for q in q_grid:
            e_kq = get_e_kq(e_k, q, nk, method='fft', R_vecs=lat.R_vecs, t_vals=lat.t_vals, k_full=finer_bz.k_full)

            chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq)
            chi_grid.append(chi0/(1 - U*chi0))
    
    else:
        q_path = (start, stop)
        k_grid = bz.k_vecs
        q_grid, path_mask = get_grid_from_path(q_path, k_grid)
        chi_grid = 1/np.asarray(invchi_grid)[path_mask]
    
    p0 = [0.01, 0., abs(1/chi_grid[len(chi_grid)//2])]
    bounds = ([0., -1., 0.], [1., 1., np.inf])

    e_hat = (stop - start)
    e_hat /= np.linalg.norm(e_hat)
    s_grid = q_grid @ e_hat
    s0 = q_min @ e_hat

    if fit_qmin:
        p0.append(s0)
        bounds[0].append(s0 - 4/nk)
        bounds[1].append(s0 + 4/nk)
        OZ_fit = OZ
    else:
        OZ_fit = lambda s, a, b, invxi: OZ(s, a, b, invxi, s0)

    try:
        par, _ = curve_fit(OZ_fit, s_grid, chi_grid, p0=p0, bounds=bounds, maxfev=10000)
        if fit_qmin:
            new_q_min = q_min - (q_min @ e_hat)*e_hat + par[3]*e_hat
        else:
            new_q_min = np.array([np.nan]*len(q_min))
            par = np.append(par, s0)
    
    except RuntimeError:
        par = np.array([np.nan]*4)
        new_q_min = np.array([np.nan]*len(q_min))

    return par, new_q_min

def density_k(e_k, S_k, mu, beta, w_k):

    Nk = len(e_k)
    if w_k is None:
        w_k = np.ones(Nk)/Nk

    x  = e_k + S_k.real - mu
    xr = np.clip(beta * x, -500, 500)
    xi = beta * S_k.imag

    cos_xi = np.cos(xi)

    ex  = np.exp(xr)
    emx = np.exp(-xr)

    nF_real = (cos_xi + emx) / (ex + 2*cos_xi + emx)

    return 2. * np.dot(nF_real, w_k)

@njit
def density_iwk(e_k, S_iwk, mu, beta, w_k, tail):

    Nk = e_k.shape[0]
    niw = S_iwk.shape[0]
    len_Sk = S_iwk.shape[1]

    if w_k is None:
        w_k = np.ones(Nk)/Nk

    n_tot = 0.0
    for k in range(Nk):
        xi = e_k[k] - mu
        Sigma_inf = S_iwk[-1, min(len_Sk - 1, k)].real
        c2 = xi + Sigma_inf

        s = 0.0 + 0.0j
        for n in range(niw):

            wn = (2*n + 1) * np.pi / beta
            iwn = 1j * wn
            Sigma = S_iwk[n, min(len_Sk - 1, k)]
            G = 1.0 / (iwn - xi - Sigma)
            if tail is True:
                G_tail = 1.0/iwn + c2/(iwn*iwn)
                s += (G - G_tail)
            else:
                s += G

        s *= (2.0 / beta)
        n_analytic = 0.5
        if tail:
            n_analytic -= (beta * c2) / 4.0
        nk = n_analytic + s.real
        n_tot += w_k[k] * nk

    return 2.0 * n_tot

def density_iwk_extr(e_k, S_iwk, mu, beta, w_k):

    niw = len(S_iwk)
    density_func = lambda n: density_iwk(e_k, S_iwk[:n+1], mu, beta, w_k)
    density = niw_extrapolate(density_func, niw)

    return density

def get_mu(e_k, n_goal, beta, niw=1, S_val=None, w_k=None, niw_extr=False):
    
    Nk = len(e_k)
    S_iwk, _ = get_iwk_arr(S_val, Nk=Nk, niw=niw)

    if w_k is None:
        w_k = np.ones(Nk)/Nk

    if niw > 1 and S_val is not None:
        if niw_extr:
            def density(mu): return density_iwk_extr(e_k, S_iwk, mu, beta, w_k, tail=False)
        else:
            def density(mu): return density_iwk(e_k, S_iwk, mu, beta, w_k, tail=True)
    else:
        def density(mu): return density_k(e_k, S_iwk[0], mu, beta, w_k)

    def f(mu): return density(mu) - n_goal

    a = e_k.min() - 50.0/beta
    b = e_k.max() + 50.0/beta
    
    return brentq(f, a, b)

def run_rpa(par, lat, bz, bz_fine=None, niw=1, S_val=None, q_path=None, method='fft', get_xi=True, xi_range=[0,0,7e-3], xi_pts=15, fit_grid_pts=True, always_fit_qmin=False, niw_fit=False, store_inputs=True, verbose=True, file_name=None):

    # Stores every input inside a dictionary
    if store_inputs:
        run_data = locals().copy()
        run_data = serialize(run_data)
    else:
        run_data = par

    if verbose:
        print('='*50)
        start_time = time.time()

    U, T, n = par['U'], par['T'], par['n']

    if verbose: print(f'U={U:.5g}, T={T:.5g}, n={n:.5g}: self-consistent mu search...', end='')

    finer_bz = bz_fine if bz_fine is not None else bz
    mu = get_mu(e_k=finer_bz.e_k, n_goal=n, beta=1/T, niw=niw, S_val=S_val, w_k=finer_bz.w_k)

    if verbose: print(f'U={U:.5g}, T={T:.5g}, n={n:.5g}: 1/chi0 minimum search over grid with nk={bz.nk}...', end='')
    Q, invchi0_Q, invchi0_grid = get_invchi0_grid(bz, mu, 1/T, q_path, method, niw, S_val, niw_fit)
    
    if bz_fine is not None:
        if verbose: print(f'U={U:.5g}, T={T:.5g}, n={n:.5g}: 1/chi0 minimum refinement with nk\'={bz_fine.nk}...', end='')
        Q, invchi0_Q = search_min(lat, bz, bz_fine, mu, 1/T, Q, q_path)
    
    invchi_Q = invchi0_Q.real - U
    run_data['invchi'] = invchi0_grid.real - U
    
    invxi = np.nan
    if get_xi:
        if invchi_Q > 0.:
            if verbose: print(f'U={U:.5g}, T={T:.5g}, n={n:.5g}: 1/xi estimation with OZ fit over chi(q)...', end='')

            fit_qmin = bz_fine is None or always_fit_qmin

            run_data['OZ_fit'], run_data['Q_fitted'] = fit_invxi(lat, bz, bz_fine, mu, 1/T, Q, U, q_path, invchi0_grid.real-U, xi_range, xi_pts, fit_grid_pts, fit_qmin)

            invxi = run_data['OZ_fit'][2]
            OZ_weight = run_data['OZ_fit'][0]

            if bz_fine is None and not np.isnan(run_data['Q_fitted'][0]):
                Q = run_data['Q_fitted']
                invchi_Q = (run_data['OZ_fit'][2]**2)/run_data['OZ_fit'][0]
                    
        else:
            run_data['OZ_fit'] = np.array([np.nan]*4)
            invxi = np.nan
            OZ_weight = np.nan
            run_data['Q_fitted'] = np.array([np.nan]*lat.dim)

    if verbose:
        Q_str = "(" + ", ".join(f"{x:.5g}" for x in Q) + ")"
        invxi_str = "None" if invxi is None else f"{invxi:.5g}"
        elapsed_time = time.time() - start_time
        print(f"U={U:.5g}, T={T:.5g}, n={n:.5g}: completed after {elapsed_time:.1f}s"
              f"results: 1/chi0 = {invchi_Q:.5g}, 1/xi = {invxi_str}, Q = {Q_str}", end='')

    run_data['invchi_min'] = invchi_Q
    run_data['Q'] = Q
    run_data['mu'] = mu
    if get_xi:
        run_data['invxi_min'] = invxi
        run_data['OZ_weight'] = OZ_weight
    else:
        run_data['invxi_min'] = np.nan
        run_data['OZ_weight'] = np.nan

    if file_name is not None:
        with HDFArchive(file_name, 'w') as ar:
            for key, val in run_data.items():
                ar[key] = val
    
    return run_data
          
def sweep_rpa(par_list, lat, bz, bz_fine=None, niw=1, S_list=None, q_path=None, method='fft', get_xi=True, xi_range=[0,0,1e-2], xi_pts=15, fit_grid_pts=True, always_fit_qmin=False, fit=False, fit_type=HMM, niw_fit=False, file_name=None, verbose=True):

    # Stores every input inside a dictionary
    sweep_data = {k: v for k, v in locals().items() if k != "par_list"}

    if verbose:
        print('='*50)
        start_time = time.time()

    # Extract the parameter list for the loop
    sweep_length = len(par_list)

    if isinstance(S_list, (list, np.ndarray)) and len(S_list) == sweep_length:
        pass
    else:
        S_list = [S_list] * sweep_length

    results_list = []
    for i, par in enumerate(par_list):
        results_list.append(run_rpa(par, lat, bz, bz_fine, niw, S_list[i], q_path, method, get_xi, xi_range, xi_pts, fit_grid_pts, 
                                    always_fit_qmin, niw_fit, store_inputs=False, verbose=False, file_name=None))
    
    merged = merge_results(results_list, ['invchi', 'invchi_min', 'invxi_min', 'Q', 'OZ_fit', 'OZ_weight', 'Q_fitted', 'mu'])
    sweep_data.update(merged)

    for key in ['U', 'T', 'n']:
        if isinstance(sweep_data[key], (list, np.ndarray)):
            varying_key = key
            varying_par = sweep_data[key]
            break

    if fit:

        if not isinstance(fit_type, (list, np.ndarray)):
            fit_type = [fit_type]
        
        sweep_data['fitchi'] = {}
        if get_xi:
            sweep_data['fitxi'] = {}

        par_keys = ['a', 'b', 'Xc']
        for label in par_keys:
            sweep_data['fitchi'][label] = []

            if get_xi:
                sweep_data['fitxi'][label] = []

        sweep_data['Qc'], sweep_data['mu_c'] = [], []

        for fit in fit_type:

            labels_to_fit = ['chi']
            if get_xi:
                labels_to_fit.append('xi')

            for label in labels_to_fit:
                pos_mask = sweep_data[f'inv{label}'] > 0
                x_fit = np.array(varying_par)[pos_mask][:10]
                y_fit = sweep_data[f'inv{label}'][pos_mask][:10]

                if len(x_fit) >= 2:
                    try:
                        p0 = [1., 1., np.min(x_fit) * 0.9]
                        bounds = ([0., 0., -np.inf], [np.inf, np.inf, np.inf])
                        par, _ = curve_fit(fit, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)

                        for i, key in enumerate(par_keys):
                            sweep_data[f'fit{label}'][key].append(par[i])
                        
                        if label == 'chi':
                            Q_vals = sweep_data['Q'][pos_mask][:2]
                            m = (Q_vals[1] - Q_vals[0])/(x_fit[1] - x_fit[0])
                            sweep_data['Qc'].append(Q_vals[0] - m * (x_fit[0] - np.maximum(0., sweep_data[f'fit{label}']['Xc'])))

                            mu_vals = sweep_data['mu'][pos_mask][:2]
                            m = (mu_vals[1] - mu_vals[0])/(x_fit[1] - x_fit[0])
                            sweep_data['mu_c'].append(mu_vals[0] - m * (x_fit[0] - np.maximum(0., sweep_data[f'fit{label}']['Xc'])))
                        
                        converged = True

                    except RuntimeError:
                        print(f"Fit of inv{label} values did not converge")
                        converged = False
                else:
                    print("Not enough points to fit!")
                    converged = False

                if not converged:
                    for i, key in enumerate(par_keys):
                        sweep_data[f'fit{label}'][key].append(np.nan)

                    if label == 'chi':
                        sweep_data['Qc'].append(np.array([np.nan]*lat.dim))
                        sweep_data['mu_c'].append(np.array([np.nan]))
                
        if len(fit_type) == 1:
            for label in labels_to_fit:
                for key in par_keys:
                    sweep_data[f'fit{label}'][key] = sweep_data[f'fit{label}'][key][0]

            sweep_data['Qc'] = sweep_data['Qc'][0]
            sweep_data['mu_c'] = sweep_data['mu_c'][0]

    if file_name is not None:
        with HDFArchive(file_name, 'w') as ar:
            for key, val in sweep_data.items():
                ar[key] = val
    
    const_keys = list(par_list[0].keys() - {varying_key})
    const_pars = [par_list[0][key] for key in const_keys]
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\r{const_keys[0]}: {const_pars[0]:.5g}, {const_keys[1]}: {const_pars[1]:.5g}: "
              f"Completed {sweep_length} jobs in {elapsed_time:.1f} seconds")

    return serialize(sweep_data)