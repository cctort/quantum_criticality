import numpy as np
from triqs.gf import GfImFreq, GfReFreq, MeshReFreq
from scipy.optimize import brentq, minimize, curve_fit
from scripts.utils import *
from scripts.lattice import *
import time
from numba import njit
import sys
from scipy.special import psi

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

        de = (e_k_eff - e_kq_eff) - 1j*(Gamma_k - Gamma_kq)

        if abs(de.real) < 1e-8:
            chi0_k = -beta * nF_k * (1.0 - nF_k)/2
        else:
            chi0_k = -(nF_k - nF_kq) / de
        
        chi0_sum += chi0_k
        
        if de_kq_dq is not None and abs(de) > 1e-8:
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
    chi0 = 0.0j
    for n in range(len(S_iwk)):
        iw = 1j * (2*n + 1) * np.pi / beta
        for k in range(Nk):
            G_iwk  = 1.0 / (iw + mu - e_k[k] - S_iwk[n,k])
            G_iwkq = 1.0 / (iw + mu - e_kq[k] - S_iwkq[n,k])
            chi0 += G_iwk * G_iwkq

    return (-2.0 / beta * chi0).real / Nk

def matsubara_rsp(beta, G_iwk, nk):

    niw = G_iwk.shape[0]
    chi0_r = np.zeros((nk,nk,nk), dtype=np.complex128)
    G_r = np.zeros((nk,nk,nk), dtype=np.complex128)

    for n in range(niw):

        G_k = G_iwk[n].reshape(nk,nk,nk)
        np.fft.ifftn(G_k, out=G_r)
        G_r_rev = np.roll(G_r[::-1, ::-1, ::-1], 1, axis=(0,1,2))
        chi0_r += G_r * G_r_rev

    chi0_q = np.fft.fftn(chi0_r)
    chi0_q *= -2.0 / beta

    invchi0_q = 1/chi0_q.real
    
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

def niw_extrapolate(invchi0_func, niw, deg=2):

    invchi0_list, niw_list = [], []
    for s in [1, 2, 3, 4]:
        n_sub = niw // s
        if n_sub < 4:
            break

        mesh_sub = np.asarray(invchi0_func(n_sub))

        invchi0_list.append(mesh_sub)
        niw_list.append(n_sub)

    x = 1. / np.array(niw_list)
    y = np.stack(invchi0_list, axis=0)

    coeff = np.polyfit(x, y, deg=deg)
    invchi0_mesh = coeff[-1].squeeze()

    return invchi0_mesh

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

    mask = (
        (dist < tol) &
        (t >= -tol) &
        (t <= length + tol)
    )

    q_grid = q_grid_full[mask]
    t_sel = t[mask]

    # sort along the path
    order = np.argsort(t_sel)
    q_grid = q_grid[order]

    # explicitly include endpoints
    if np.linalg.norm(q_grid[0] - start) > tol:
        q_grid = np.vstack([start, q_grid])

    if np.linalg.norm(q_grid[-1] - stop) > tol:
        q_grid = np.vstack([q_grid, stop])

    return q_grid

def get_invchi0_min(lat, mu, beta, nk, q_path=None, method='matsubara', niw=1, S_iwk=None, refine=True, refine_ratio=1., niw_extr=True, ibz=True):
    
    dim = lat.dim
    e_k = lat.e_k
    Nk = len(e_k)

    # q_grid either from q_path or from the BZ/IBZ
    if q_path is not None:
        q_grid = get_grid_from_path(q_path, lat.k_vecs)
    else:
        q_grid = lat.k_vecs / np.pi

    # From S_val to (niw, Nk) array
    S_iwk, k_dep = get_iwk_arr(S_iwk, Nk, niw)

    if method == 'lindhard':

        # Unfolding e_k and S_iwk into the full BZ
        if ibz:
            e_k = lat.unfold_f_k(e_k)
            if k_dep:
                S_iwk = lat.unfold_f_iwk(S_iwk)
        
        # 1/chi0 for each q using Lindhard
        invchi0_grid = []
        for q in q_grid:
            e_kq = lat.get_e_kq(e_k, q, nk)

            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, nk)
            else:
                S_iwkq = S_iwk

            chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_iwkq[0])
            invchi0_grid.append(1/chi0)

    elif method == 'matsubara':

        # Unfold e_k before G_iwk evaluation (faster if we do not have to unfold f_iwk also)
        if ibz and not k_dep:
            e_k = lat.unfold_f_k(e_k)

        G_iwk = np.array(get_G_iwk(mu, beta, e_k, S_iwk, niw))

        # if not already unfolded, unfold directly G_iwk
        if ibz and k_dep:
            G_iwk = lat.unfold_f_iwk(G_iwk)
        
        # Evaluate 1/chi0 over whole BZ using FFT back and forth
        if niw_extr:
            invchi0_func = lambda niw: matsubara_rsp(beta, G_iwk[:niw+1], nk)
            invchi0_grid = niw_extrapolate(invchi0_func, niw)
        else:
            invchi0_grid = matsubara_rsp(beta, G_iwk, nk)
        
        del G_iwk

        # Fold into IBZ to save memory
        if ibz:
            invchi0_grid = lat.fold_f_k(invchi0_grid)

    # Minimum of 1/chi0 over q_grid
    idx = np.argmin(invchi0_grid)
    q_min = q_grid[idx]
    invchi0_min = invchi0_grid[idx]

    # Further minimum search over finer grid + gradient descent type method
    if refine:

        # Get new nk, e_k and S_iwk
        nk_fine = int(refine_ratio * nk)

        e_k = lat.e_k_fine
        if ibz:
            e_k = lat.unfold_f_k(e_k, fine=True)

        if k_dep:
            if ibz:
                S_iwk = lat.unfold_f_iwk(S_iwk, fine=True)

            # If k_dep, precalculate S_iwR for the FFT
            S_iwR = lat.get_f_iwR(S_iwk, nk_fine, fine=True)

        # Evaluates e_kq/S_iwkq shifting the arrays by q with np.roll
        def invchi0_q_roll(s):
            q = start + np.dot(J, np.atleast_1d(s))

            e_kq = lat.get_e_kq(e_k, q, nk_fine)
            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, nk_fine)
            else:
                S_iwkq = S_iwk

            chi0, _ = lindhard_ksp(mu, beta, e_k, e_kq, S_iwk[0], S_iwkq[0])

            return 1/chi0

        # Evaluates e_kq/S_iwkq with the exact formulas, requires t_vals and S_iwR
        def invchi0_q_exact(s):
            q = start + np.dot(J, np.atleast_1d(s))

            e_kq, de_kq_dq = lat.get_e_kq(e_k, q, nk_fine, method='exact')
            if k_dep:
                S_iwkq = lat.get_f_iwkq(S_iwk, q, nk_fine, method='exact', f_iwR=S_iwR)
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
            step  = 1 / (len(q_grid) - 1)
            s0    = np.array([idx / (len(q_grid) - 1)])
            lo    = max(0.0, s0[0] - safe*step)
            hi    = min(1.0, s0[0] + safe*step)
            bounds = [(lo, hi)]

        else:
            start = q_grid[0]
            J = lat.b_vecs / (4*np.pi) #
            J_inv = np.linalg.inv(J)
            step = 1.0 / nk
            s0 = J_inv @ (q_min - start)
            bounds = []
            for d in range(dim):
                lo = max(0.0, s0[d] - safe*step)
                hi = min(1.0, s0[d] + safe*step)
                bounds.append((lo, hi))

        if refine_ratio > 1.:

            if q_path is not None:
                q_grid_fine = get_grid_from_path(q_path, lat.k_vecs_fine)
                start = q_grid_fine[0]
                J_inv = np.linalg.pinv(J)
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
                step  = 1.0 / nk_fine
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

    q_min = 1 - np.sort(np.abs(q_min - 1))

    return q_min, invchi0_min, invchi0_grid

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
    return n_tot / np.sum(ibz_w_k)

def get_mu(e_k, n_goal, beta, niw=1, S_iwk=None, ibz_w_k=None):
    
    Nk_ibz = len(e_k)

    S_iwk, k_dep = get_iwk_arr(S_iwk, Nk=Nk_ibz, niw=niw)

    if ibz_w_k is None:
        ibz_w_k = np.ones(Nk_ibz)

    if k_dep:
        def density(mu): return density_iwk(e_k, S_iwk, mu, beta, ibz_w_k)
    else:
        def density(mu): return density_k(e_k, S_iwk[0], mu, beta, ibz_w_k)

    def f(mu): return density(mu) - n_goal

    # temperature-aware bracketing
    a = e_k.min() - 50.0/beta
    b = e_k.max() + 50.0/beta
    
    return brentq(f, a, b)
          
def sweep_chirpa(par_dict, t=1., tp=0., dim=3, nk=100, niw=1, S_iwk_list=None, q_path=None, method='matsubara', refine=True, refine_ratio=1., nk_avg=(1,1), fit=False, fit_type=critical1, fix_exp=None, niw_extr=True, ibz=True, save_file=None, verbose=True):
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
            lat.get_bz(nk_fine, ibz=ibz, fine=True)
            lat.get_e_k(fine=True)

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

        for key in ['ampl', 'gamma', 'offset', 'Xc', 'Qc', 'mu_c',
                    'ampl_err', 'gamma_err', 'Xc_err', 'offset_err']:
            results[key] = []

        for fit in fit_type:
            if len(x_fit) >= 2:
                try:
                    
                    p0 = [1., 1., np.min(x_fit) * 0.9]
                    bounds = ([0., 0., -np.inf],
                            [np.inf, np.inf, np.inf])
                    par_labels = ['ampl', 'gamma', 'offset']

                    fit_new = fit


                    par_fit, cov_fit = curve_fit(
                        fit_new,
                        x_fit,
                        y_fit,
                        p0=p0,
                        bounds=bounds,
                        maxfev=10000
                    )

                    par = par_fit
                    cov = cov_fit
                    for i, label in enumerate(par_labels):
                        results[label].append(par[i])
                        err = np.sqrt(cov[i, i])
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