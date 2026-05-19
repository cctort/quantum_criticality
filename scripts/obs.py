import numpy as np
from triqs.gf import GfImFreq, GfReFreq, MeshReFreq
from scipy.optimize import brentq, minimize, curve_fit, root_scalar
from scripts.utils import *
from scripts.lattice import *
import time
from numba import njit

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
def lindhard_ksp(mu, beta, e_k, e_kq, S_k, S_kq, de_kq_dq=None, ibz_w_k=None):
   
    Nk = len(e_k)
    chi0_sum = 0.0 + 0.0j
    
    if ibz_w_k is None:
        ibz_w_k = np.ones(len(e_k))

    if de_kq_dq is not None:
        dim = de_kq_dq.shape[0]
        dchi0_sum = np.zeros(dim)

    lenS_k = len(S_k)
    lenS_kq = len(S_kq)

    for k in range(Nk):
        s_k = S_k[min(lenS_k - 1, k)]
        s_kq = S_kq[min(lenS_kq - 1, k)]

        e_k_eff = e_k [k] + s_k.real
        e_kq_eff = e_kq[k] + s_kq.real
        Gamma_k = -s_k.imag
        Gamma_kq = -s_kq.imag

        xk = beta * (e_k_eff + 1j*Gamma_k - mu)
        xkq = beta * (e_kq_eff + 1j*Gamma_kq - mu)

        if xk.real > 500.0:
            nF_k = 0.0 + 0.0j
        elif xk.real < -500.0:
            nF_k = 1.0 + 0.0j
        else:
            nF_k = 1.0 / (cexp(xk) + 1.0)

        if xkq.real > 500.0:
            nF_kq = 0.0 + 0.0j
            dnF_kq = 0.0 + 0.0j
        elif xkq.real < -500.0:
            nF_kq = 1.0 + 0.0j
            dnF_kq = 0.0 + 0.0j
        else:
            ekq_val = cexp(xkq)
            nF_kq = 1.0 / (ekq_val + 1.0)
            dnF_kq = -ekq_val / (ekq_val + 1.0)**2

        de = (e_k_eff - e_kq_eff) - 1j*(Gamma_k + Gamma_kq)

        if abs(de) < 1e-8:
            chi0_k = -beta * nF_k * (1.0 - nF_k)
        else:
            chi0_k = -(nF_k - nF_kq) / de
        
        chi0_sum += chi0_k * ibz_w_k[k]
        
        if de_kq_dq is not None and abs(de) > 1e-8:
            for alpha in range(dim):
                v = de_kq_dq[alpha, k]
                num = (dnF_kq * beta * v) * de - (nF_k - nF_kq) * v
                dchi0_sum[alpha] += (num / de**2).real * ibz_w_k[k]

    chi0 = chi0_sum.real / Nk

    if de_kq_dq is None:
        dchi0 = None

    else: 
        dchi0 = np.zeros(dim)
        for alpha in range(dim):
            dchi0[alpha] = dchi0_sum[alpha] / Nk
    
    return chi0, dchi0

@njit
def matsubara_ksp(mu, beta, e_k, e_kq, S_iwk, S_iwkq, weights):
    
    Nk = len(e_k)
    chi0 = 0.0 + 0.0j
    for n in range(len(S_iwk)):
        iw = 1j * (2*n + 1) * np.pi / beta
        for k in range(Nk):
            G_iwk  = 1.0 / (iw + mu - e_k[k] - S_iwk[n,k])
            G_iwkq = 1.0 / (iw + mu - e_kq[k] - S_iwkq[n,k])
            chi0 += G_iwk * G_iwkq * weights[k]

    return (-2.0 / beta * chi0).real / Nk

def matsubara_rsp(mu, beta, e_k, S_iwk, nk, niw):

    ek_mesh = e_k.reshape(nk,nk,nk)
    uniform = np.allclose(S_iwk, S_iwk[0,0])

    iw_n  = (2*np.arange(niw) + 1) * np.pi / beta
    chi0_r = np.zeros((nk,nk,nk), dtype=np.complex128)
    G_r    = np.zeros((nk,nk,nk), dtype=np.complex128)

    for n in range(niw):
        if not uniform:
            S_val = S_iwk[n, 0] if S_iwk.shape[1] == 1 else S_iwk[n].reshape(nk,nk,nk)
        else:
            S_val = S_iwk[0,0]

        G_k = 1.0 / (1j*iw_n[n] + mu - ek_mesh - S_val)
        np.fft.ifftn(G_k, out=G_r)
        del G_k

        chi0_r.real += G_r.real * G_r.real - G_r.imag * G_r.imag
        chi0_r.imag += 2.0 * G_r.real * G_r.imag

    chi0_q = np.fft.fftn(chi0_r)
    chi0_q *= -2.0 / beta
    invchi0_mesh = chi0_q.real.copy()
    np.reciprocal(invchi0_mesh, out=invchi0_mesh)
    
    return invchi0_mesh.reshape(-1)

def niw_extrapolate(invchi0_func, niw, deg=2):

    invchi0_list, niw_list = [], []
    for s in [1, 2, 3, 4]:
        n_sub = niw // s
        if n_sub < 4:
            break

        mesh_sub = np.asarray(invchi0_func(n_sub))

        invchi0_list.append(mesh_sub)
        niw_list.append(n_sub)

    x = 1.0 / np.array(niw_list)
    y = np.stack(invchi0_list, axis=0)

    coeff = np.polyfit(x, y, deg=deg)
    invchi0_mesh = coeff[-1].squeeze()

    return invchi0_mesh

def get_iwk_arr(S_val, Nk, dim, niw=1):
    k_dep = False

    if S_val is None or isinstance(S_val, (int, float, complex)):
        val = 0j if S_val is None else complex(S_val)
        # if caller wants niw > 1 with a constant sigma, broadcast over frequencies too
        S_val = np.full((niw, 1), val, dtype=complex)

    else:
        S_val = np.asanyarray(S_val, dtype=complex)

        if S_val.ndim == 1:
            if len(S_val) == Nk:
                # k-dependent, no frequency axis — replicate to niw if requested
                S_val = np.tile(S_val[np.newaxis, :], (niw, 1))  # (niw, Nk)
                k_dep = True
            else:
                # k-independent, frequency-dependent: (len,) → (len, 1)
                # input length takes priority over niw
                niw = len(S_val)
                S_val = S_val[:, np.newaxis]                        # (niw, 1)

        elif S_val.ndim == dim:
            # k-dependent spatial array, no frequency axis
            S_val = np.tile(S_val.reshape(1, Nk), (niw, 1))      # (niw, Nk)
            k_dep = True

        else:
            # (niw', Nk) already — niw' takes priority
            niw = S_val.shape[0]
            k_dep = True

    return S_val, niw, k_dep

def expand(S, Nk):
    return S if S.shape[1] > 1 else np.tile(S, (1, Nk))

def get_grid_from_path(q_path, k_grid):

    start = np.array(q_path[0])
    stop = np.array(q_path[-1])
    q_grid_full = k_grid / np.pi

    direction = stop - start
    norm_sq = np.dot(direction, direction)

    diff = q_grid_full - start
    t = diff @ direction / norm_sq

    residuals = diff - np.outer(t, direction)
    on_line = np.linalg.norm(residuals, axis=1) < 1e-10
    in_range = (t >= 0) & (t <= 1)

    mask = on_line & in_range
    q_grid = q_grid_full[mask]

    return q_grid[np.argsort(t[mask])]

def get_invchi0_min(lat, mu, beta, nk, q_path=None, method='matsubara', niw=1, S_iwk=None, refine=True, refine_ratio=1., niw_extr=True, ibz=False):
    dim = lat.dim
    e_k = lat.e_k
    Nk = len(e_k)

    S_iwk, niw, k_dep = get_iwk_arr(S_iwk, nk**dim, dim, niw)

    if q_path is not None:
        q_grid = get_grid_from_path(q_path, lat.full_k_vecs)
    else:
        q_grid = lat.k_vecs / np.pi

    if method == 'lindhard':

        if k_dep:
            S_k = expand(S_iwk, Nk)
        else:
            S_k = [[S_iwk[0,0]]]

        S_iwkq = np.empty_like(S_iwk)
        
        invchi0_grid = []
        for q in q_grid:
            e_kq = lat.get_e_kq(e_k, q, nk, ibz)
            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, nk, ibz)
                S_kq = S_iwkq
            else:
                S_kq = S_k

            chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq, S_k[0], S_kq[0], ibz_w_k=lat.ibz_w_k)
            invchi0_grid.append(1/chi0)

    elif method == 'matsubara':#to do ibz
        if niw_extr:
            invchi0_func = lambda niw: matsubara_rsp(mu, beta, e_k, S_iwk, nk, niw)
            invchi0_grid = niw_extrapolate(invchi0_func, niw)
        else:
            invchi0_grid = matsubara_rsp(mu, beta, e_k, S_iwk, nk, niw)

    idx = np.argmin(invchi0_grid)
    q_min = q_grid[idx]
    invchi0_min = invchi0_grid[idx]

    if refine:
        nk_fine = int(refine_ratio * nk)
        e_k = lat.e_k_fine

        if k_dep:
            S_iwR = lat.get_f_iwR(S_iwk, nk_fine)

        S_iwkq = np.empty_like(S_iwk) if k_dep else None

        def invchi0_q_coarse(s):
            q = start + np.dot(J, np.atleast_1d(s))
            e_kq = lat.get_e_kq(e_k, q, nk_fine, ibz)
            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, nk_fine, ibz)
                S_kq = S_iwkq[0]
            else:
                S_kq = S_iwk[0]

            chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_kq, lat.ibz_w_k_fine)

            return 1/chi0

        def invchi0_q_fine(s):
            q = start + np.dot(J, np.atleast_1d(s))
            e_kq, de_kq_dq = lat.get_e_kq(e_k, q, nk_fine, ibz, 'fine')
            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, nk_fine, ibz, 'fine', f_iwR=S_iwR)
                S_kq = S_iwkq[0]
            else:
                S_kq = S_iwk[0]

            chi0, dchi0_dq = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_kq, de_kq_dq=de_kq_dq, ibz_w_k=lat.ibz_w_k_fine)
            dinvchi0_dq = -dchi0_dq / chi0**2
            grad_s = J.T @ dinvchi0_dq

            return 1/chi0, grad_s
        
        safe = 1

        if q_path is not None:
            start = q_grid[0]
            stop = q_grid[-1]
            J     = (stop - start).reshape(dim, 1)
            step  = 1 / (len(q_grid) - 1)
            s0    = np.array([idx / (len(q_grid) - 1)])
            lo    = max(0.0, s0[0] - safe*step)
            hi    = min(1.0, s0[0] + safe*step)
            bounds = [(lo, hi)]

        else:
            grid_shape = (nk,) * dim
            start = q_grid[0]
            idx   = np.unravel_index(idx, grid_shape)
            J     = np.zeros((dim, dim))
            for d in range(dim):
                idx1    = [0] * dim
                idx1[d] = 1
                J[:, d] = (q_grid[np.ravel_multi_index(idx1, grid_shape)] - q_grid[0]) * nk
            step  = 1.0 / nk
            s0    = np.array(idx, dtype=float) / nk
            bounds = []
            for d in range(dim):
                lo = max(0.0, s0[d] - safe*step)
                hi = min(1.0, s0[d] + safe*step)
                bounds.append((lo, hi))

        if refine_ratio > 1.:

            if q_path is not None:
                q_grid_fine = get_grid_from_path(q_path, lat.k_vecs_fine)
                start = q_grid_fine[0]
                J_pinv = np.linalg.pinv(J)
                s_grid_fine = np.array([J_pinv @ (q - start) for q in q_grid_fine]).squeeze()
                mask = (s_grid_fine >= lo) & (s_grid_fine <= hi)
                s_grid = s_grid_fine[mask]

                step  = 1 / (len(q_grid_fine) - 1)
                lo    = max(0.0, s0[0] - safe*step)
                hi    = min(1.0, s0[0] + safe*step)
                bounds = [(lo, hi)]
            else:
                q_grid_fine = lat.k_vecs_fine / np.pi
                J_inv = np.linalg.inv(J)
                s_grid_fine = np.array([J_inv @ (q - start) for q in q_grid_fine])
                mask = np.all((s_grid_fine >= [b[0] for b in bounds]) & 
                            (s_grid_fine <= [b[1] for b in bounds]), axis=1)
                s_grid = s_grid_fine[mask]

                step  = 1.0 / nk_fine
                bounds = []
                for d in range(dim):
                    lo = max(0.0, s0[d] - safe*step)
                    hi = min(1.0, s0[d] + safe*step)
                    bounds.append((lo, hi))
            
            invchi0_grid_fine = np.array([invchi0_q_coarse(s) for s in s_grid])
            s0 = np.atleast_1d(s_grid[np.argmin(invchi0_grid_fine)])

        res = minimize(
            invchi0_q_fine,
            x0=s0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'ftol': 1e-6, 'gtol': 1e-4, 'maxiter': 50}
        )
        s_min = res.x

        q_min = start + J @ s_min
        invchi0_min, _ = invchi0_q_fine(s_min)

    return q_min, invchi0_min, invchi0_grid

def density_k(e_k, S_k, mu, beta, ibz_w_k):
    x  = e_k + S_k.real - mu
    xr = np.clip(beta * x, -500, 500)
    xi = beta * S_k.imag

    cos_xi = np.cos(xi)

    ex  = np.exp(xr)
    emx = np.exp(-xr)

    nF_real = (cos_xi + emx) / (ex + 2*cos_xi + emx)

    return 2. * np.dot(nF_real, ibz_w_k)*len(e_k)

@njit
def density_iwk(e_k, S_iwk, mu, beta, ibz_w_k):
    niw, Nk = S_iwk.shape
    n_tot = 0.0
    for k in range(Nk):
        e = e_k[k]
        xi_k = e - mu
        g_sum = 0.0
        g0_sum = 0.0
        for n in range(niw):
            w = (2*n + 1) * np.pi / beta
            Sigma = S_iwk[n, k]
            denom_r = mu - e - Sigma.real
            denom_i = w - Sigma.imag
            g_sum  += denom_r / (denom_r**2 + denom_i**2)
            g0_sum += (-xi_k)  / (w**2    + xi_k**2)
        
        x = beta * xi_k
        if x > 500.0:
            nF = 0.0
        elif x < -500.0:
            nF = 1.0
        else:
            nF = 1.0 / (np.exp(x) + 1.0)
        g_corr = g_sum - g0_sum + nF
        n_tot += 2.0/beta * g_corr * ibz_w_k[k]
    return n_tot / Nk

def get_mu(e_k, n_goal, beta, niw=1, S_iwk=None, dim=None, ibz_w_k=None):
    
    S_iwk, niw, k_dep = get_iwk_arr(S_iwk, Nk=len(e_k), dim=dim, niw=niw)

    if ibz_w_k is None:
        ibz_w_k = np.ones(len(e_k))

    if k_dep:
        def density(mu): return density_iwk(e_k, S_iwk, mu, beta, ibz_w_k)
    else:
        def density(mu): return density_k(e_k, S_iwk[0], mu, beta, ibz_w_k)

    def f(mu): return density(mu) - n_goal

    # temperature-aware bracketing
    a = e_k.min() - 50.0/beta
    b = e_k.max() + 50.0/beta
    
    return brentq(f, a, b)
          
def sweep_chirpa(par_dict, t=1., tp=0., dim=3, nk=100, niw=1, S_iwk_list=None, q_path=None, method='matsubara', refine=True, refine_ratio=1., nk_avg=(1,1), fit=False, fit_type=critical1, fix_exp=None, niw_extr=True, ibz=False, save_file=None, verbose=True):
    """
    Run a 1D sweep of DMFT calculations over a list of tuples (T, U or n).
    """

    start_time = time.time()

    lat = LATTICE(t=t, tp=tp, dim=dim)

    # Extract the parameter list for the loop
    list_label = next(k for k, v in par_dict.items() if isinstance(v, (list, np.ndarray)))

    if isinstance(S_iwk_list, (list, np.ndarray)) and len(S_iwk_list) == len(par_dict[list_label]):
        pass
    else:
        S_iwk_list = [S_iwk_list] * len(par_dict[list_label])
    
    # Sweep initialization
    results = {'t': lat.t, 'dim': lat.dim, 'nk': nk, 'niw': niw,
               'tp': tp, 'method': method, 'q_path': q_path, 'nk_avg': nk_avg,
               'S_iwk_list': S_iwk_list, 'refine': refine, 'refine_ratio': refine_ratio,
               'niw_extr': niw_extr, 'invchi': None,  'Q': None, 'mu': None, 'par_list': []}

    nk_list = [nk - i*nk_avg[1] for i in range(nk_avg[0])]
    Q_avg = np.zeros((len(par_dict[list_label]), 3))
    invchi0_avg = np.zeros(len(par_dict[list_label]))
    mu_avg = np.zeros(len(par_dict[list_label]))
    total_jobs = len(par_dict[list_label])*len(nk_list)
    for i_nk, nk_val in enumerate(nk_list):
        
        lat.get_bz(nk, ibz=ibz)
        lat.get_e_k()

        if refine:
            nk_fine = int(refine_ratio*nk_val)
            if refine_ratio > 1:
                lat.get_bz(nk_fine, ibz=ibz, fine=True)
                lat.get_e_k(fine=True)
                lat.get_phase_k()
            else:
                lat.k_vecs_fine = lat.k_vecs
                lat.e_k_fine = lat.e_k
                lat.full_k_vecs_fine = lat.full_k_vecs
                lat.ibz_w_k_fine = lat.ibz_w_k
                lat.get_phase_k()

        # Parameter sweep
        for i_var, var in enumerate(par_dict[list_label]):

            par = {**par_dict, list_label: var}
            U, T, n = par['U'], par['T'], par['n']

            finer_e_k = lat.e_k_fine if refine else lat.e_k
            finer_ibz_w_k = lat.ibz_w_k_fine if refine else lat.ibz_w_k
            mu = get_mu(e_k=finer_e_k, n_goal=n, beta=1/T, S_iwk=S_iwk_list[i_var], ibz_w_k=finer_ibz_w_k)
            mu_avg[i_var] += mu

            Q, invchi0_Q, _ = get_invchi0_min(lat, mu, beta=1/T, nk=nk_val, q_path=q_path, S_iwk=S_iwk_list[i_var], method=method, niw=niw, refine=refine, refine_ratio=refine_ratio, niw_extr=niw_extr, ibz=ibz)

            # Results for each U,T,n point
            invchi0_avg[i_var] += invchi0_Q.real
            Q_avg[i_var] += np.array(Q)

            if i_nk == 0:
                results['par_list'].append({'U': U, 'T': T, 'n': n})

            if verbose:
                job = i_var+i_nk*len(par_dict[list_label])+1
                print(f"\rJob {job}/{total_jobs}: U={U:.3f}, T={T:.3f}, n={n:.3f}", end='')

    results['invchi'] = invchi0_avg / len(nk_list) - U
    results['Q'] = Q_avg / len(nk_list)
    results['mu'] = mu_avg / len(nk_list)

    if fit:
        if not isinstance(fit_type, (list, np.ndarray)):
            fit_type = [fit_type]
            fix_exp = [fix_exp]
        else:
            if fix_exp is None:
                fix_exp = [None]*len(fit_type)

        pos_mask = results['invchi'] > 0
        x_fit = np.array(par_dict[list_label])[pos_mask][:15]
        y_fit = results['invchi'][pos_mask][:15]

        for key in ['ampl', 'gamma', 'c', 'd', 'Xc', 'Qc', 'mu_c',
                    'ampl_err', 'gamma_err', 'Xc_err', 'c_err', 'd_err']:
            results[key] = []

        for i_fit, fit in enumerate(fit_type):
            if len(x_fit) >= 2:
                try:
                    
                    if fit == critical3:
                        p0 = [1., 1., np.min(y_fit) * 0.9, 0.1]
                        bounds = ([0., 0., -np.inf, 1e-12],
                                [np.inf, np.inf, np.inf, 1.])
                        par_labels = ['ampl', 'gamma', 'c', 'd']
                    else:
                        p0 = [1., 1., np.min(x_fit) * 0.9]
                        bounds = ([0., 0., -np.inf],
                                [np.inf, np.inf, np.inf])
                        par_labels = ['ampl', 'gamma', 'c']

                    fit_new = fit

                    if fit == critical3 and fix_exp[i_fit] is not None:

                        gamma_fix = fix_exp[i_fit]

                        fit_new = lambda x, a, c, d: critical3(x, a, gamma_fix, c, d)

                        p0.pop(1)

                        bounds = (list(bounds[0]), list(bounds[1]))
                        bounds[0].pop(1)
                        bounds[1].pop(1)

                    par_fit, cov_fit = curve_fit(
                        fit_new,
                        x_fit,
                        y_fit,
                        p0=p0,
                        bounds=bounds,
                        maxfev=100000
                    )

                    par = par_fit
                    cov = cov_fit

                    if fit == critical3 and fix_exp[i_fit] is not None:

                        par = np.insert(par_fit, 1, gamma_fix)

                        cov_full = np.zeros((len(par), len(par)))
                        mask = [i for i in range(len(par)) if i != 1]

                        for i_old, i_new in enumerate(mask):
                            for j_old, j_new in enumerate(mask):
                                cov_full[i_new, j_new] = cov_fit[i_old, j_old]

                        cov = cov_full

                    for i, label in enumerate(par_labels):

                        results[label].append(par[i])

                        err = 0. if (fit == critical3 and fix_exp[i_fit] is not None and i == 1) else np.sqrt(cov[i, i])

                        results[f'{label}_err'].append(err)

                    if fit == critical1:

                        a, b, c = par

                        z = -c / a
                        Xc = np.sign(z) * np.abs(z)**(1 / b)

                        dXc_dz = (1 / b) * np.abs(z)**(1 / b - 1)

                        grad = np.array([
                            -dXc_dz * z / a,
                            -Xc * np.log(np.abs(z) + 1e-15) / b**2,
                            -dXc_dz / a
                        ])

                    elif fit == critical2:

                        Xc = par[2]
                        grad = np.array([0., 0., 1.])

                    elif fit == critical3:

                        def froot(x):
                            return critical3(x, par[0], par[1], par[2], par[3])

                        x0 = np.median(x_fit)

                        try:
                            sol = root_scalar(
                                froot,
                                x0=x0,
                                x1=x0 * 1.1 + 1e-12,
                                method='secant'
                            )
                            Xc = sol.root

                        except ValueError:

                            xmin = max(np.min(x_fit), 1e-12)
                            xmax = np.max(x_fit)

                            if froot(xmin) * froot(xmax) > 0:
                                Xc = np.nan
                            else:
                                sol = root_scalar(
                                    froot,
                                    bracket=[xmin, xmax],
                                    method='brentq'
                                )
                                Xc = sol.root

                        eps = 1e-8

                        def dfdp(params, idx):
                            p_plus = params.copy()
                            p_minus = params.copy()
                            p_plus[idx] += eps
                            p_minus[idx] -= eps
                            return (critical3(Xc, *p_plus) - critical3(Xc, *p_minus)) / (2 * eps)

                        grad = np.array([dfdp(par, i) for i in range(len(par))])

                    Xc_err = np.sqrt(np.abs(grad @ cov @ grad))

                    results['Xc'].append(Xc)
                    results['Xc_err'].append(Xc_err)

                    Q_vals = results['Q'][pos_mask][:2]
                    m = (Q_vals[1] - Q_vals[0])/(x_fit[1] - x_fit[0])
                    results['Qc'].append(Q_vals[0] - m * (x_fit[0] - Xc))

                    mu_vals = results['mu'][pos_mask][:2]
                    m = (mu_vals[1] - mu_vals[0])/(x_fit[1] - x_fit[0])
                    results['mu_c'].append(mu_vals[0] - m * (x_fit[0] - Xc))

                except RuntimeError as e:
                    print(f"\nFit failed: {e}")
                    
                    for key in ['Xc', 'gamma', 'ampl', 'Xc_err', 'gamma_err', 'ampl_err']:
                        results[key].append(0.)
                    
                    results['Qc'].append(np.array([np.nan]*dim))
                    results['mu_c'].append(np.nan)
                    
            else:
                print("\nFit skipped: fewer than 2 positive points")
                
                for key in ['Xc', 'gamma', 'ampl', 'Xc_err', 'gamma_err', 'ampl_err']:
                    results[key].append(0.)
                    
                results['Qc'].append(np.array([np.nan]*dim))
                results['mu_c'].append(np.nan)

        if len(fit_type) == 1:
            for key in ['Xc', 'gamma', 'ampl', 'Xc_err', 'gamma_err', 'ampl_err', 'Qc', 'mu_c']:
                results[key] = results[key][0]

    if save_file is not None:
        HDFwrite_dict(save_file, results)
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\rCompleted {len(par_dict[list_label])} jobs in {elapsed_time:.2f} seconds | U={U:.3f}, T={T:.3f}, n={n:.3f}")

    return results