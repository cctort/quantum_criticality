import numpy as np
from triqs.gf import *
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
from triqs_tprf.lattice import lattice_dyson_g0_wk, lattice_dyson_g_wk, lindhard_chi00
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scripts.utils import *
from scripts.lattice import *
import time
from scipy.optimize import curve_fit
from numba import njit


def get_Z(S_iw_data, beta, n_iw):
    """
    Evaluate quasi-particle weight Z, scattering rate gamma, and lifetime tau from self-energy.
    """
    
    n = np.arange(n_iw)
    iw = (2*n + 1) * np.pi / beta

    iw0 = iw[0]
    S0 = S_iw_data[n_iw].imag

    Z = 1. / (1. - S0 / iw0)
    gamma = -Z * S0
    tau = 1. / gamma

    return Z, gamma, tau

def get_invchi0(mu, e_k, iw_mesh, verbose=False, save_memory=False):
    """
    Evaluate inverse bare susceptibility chi0.
    """
    
    #G0_iwk = lattice_dyson_g0_wk(mu, e_k, iw_mesh)

    n_iw = len(iw_mesh)
    #chi0_iwk = imtime_bubble_chi0_wk(G0_iwk, nw=n_iw, verbose=verbose, save_memory=save_memory)
    chi0_iwk = lindhard_chi00(e_k=e_k, mesh=iw_mesh, mu=mu)
    chi0_iwk.data[...] = 1. / chi0_iwk.data

    return chi0_iwk

def get_invchi0_fromSiw(mu, e_k, S_iw, verbose=False):
    """
    Evaluate inverse bare susceptibility chi0.
    """
    
    G0_iwk = lattice_dyson_g_wk(mu, e_k, S_iw)

    n_iw = len(S_iw.data[:,0,0])
    chi0_iwk = imtime_bubble_chi0_wk(G0_iwk, nw=n_iw, verbose=verbose)
    chi0_iwk.data[...] = 1. / chi0_iwk.data

    return chi0_iwk

#def get_invchi0_dressed(mu, eps_k, method='bare', iw_mesh=None, S_iw=None, verbose=False):
    #n_iw = len(S_iw.mesh)
    #G_iwk = lattice_dyson_g_wk(mu, eps_k, S_iw)
    #chi0_iwk = imtime_bubble_chi0_wk(G_iwk, nw=n_iw, verbose=verbose)

def lindhard(e_k, e_kq, mu, beta):
    
    def fermi(eps):
        x = beta * (eps - mu)
        return np.where(x >  500, 0.0,
               np.where(x < -500, 1.0,
               1.0 / (np.exp(np.clip(x, -500, 500)) + 1.0)))

    nF     = fermi(e_k)
    nF_kpq = fermi(e_kq)
    de     = e_k - e_kq

    mask    = np.abs(de) < 1e-8
    safe_de = np.where(mask, 1.0, de)
    chi0    = np.where(mask, -beta * nF * (1 - nF), -(nF - nF_kpq) / safe_de)

    return np.mean(chi0)

@njit
def matsubara_sum(e_k, e_kq, mu, beta, S_iw_arr):
    N_k = len(e_k)
    chi0 = 0.0 + 0.0j
    for n in range(len(S_iw_arr)):
        iw = 1j * (2*n + 1) * np.pi / beta
        for k in range(N_k):
            G_iwk  = 1.0 / (iw + mu - e_k[k] - S_iw_arr[n])
            G_iwkq = 1.0 / (iw + mu - e_kq[k] - S_iw_arr[n])
            chi0 += G_iwk * G_iwkq
    return (-2.0 / beta * chi0).real / N_k

def get_min_invchi0(lattice, mu, beta, q_path=None, S_iw_arr=None, refine=True, n_iw_extr=True):

    invchi_grid = []
    for q in q_path:
        e_k, e_kq = lattice.get_e_kq(q, 'coarse')

        if S_iw_arr is None:
            invchi_grid.append(1/lindhard(e_k, e_kq, mu, beta))
        else:
            invchi_grid.append(1/matsubara_sum(e_k, e_kq, mu, beta, S_iw_arr))
    
    idx = np.argmin(invchi_grid)
    best_q = q_path[idx]
    invchi0_min = invchi_grid[idx]

    if refine and q_path is not None:
        
        start = q_path[0]
        end = q_path[-1]
        step = 1 / (len(q_path) - 1)

        def invchi0(s, S_iw_arr):
            
            q = start + s * (end - start)
            e_k, e_kq = lattice.get_e_kq(q, 'fine')

            if S_iw_arr is None:
                return 1/lindhard(e_k, e_kq, mu, beta)
            else:
                return 1/matsubara_sum(e_k, e_kq, mu, beta, S_iw_arr)

        s0 = idx / (len(q_path) - 1)

        res = minimize_scalar(invchi0, method='bounded', bounds=(max(0, s0 - step), min(1, s0 + step)), args=(S_iw_arr,))
        
        s_min = res.x
        best_q = start + s_min * (end - start)
        invchi0_min = invchi0(s_min, S_iw_arr)
    
    if n_iw_extr and S_iw_arr is not None:
        
        n_iw_list = [len(S_iw_arr)]
        invchi0_min_list = [invchi0_min]

        subs = [2, 3, 4]

        for s in subs:

            n_iw = len(S_iw_arr)//s
            n_iw_list.append(n_iw)
            invchi0_min_list.append(invchi0(s_min, S_iw_arr[:n_iw+1]))

        coeff = np.polyfit(1./np.array(n_iw_list), invchi0_min_list, 2)
        invchi0_min = coeff[-1]

    return best_q, invchi0_min


def get_mu(e_k, n_goal, beta):
    eps = e_k.data[..., 0, 0].real.flatten()
    def get_n(mu):
        x = np.clip(beta * (eps - mu), -500, 500)
        return np.mean(2. / (np.exp(x) + 1.)) - n_goal
    return brentq(get_n, eps.min() - 10, eps.max() + 10)

          
def sweep_chirpa(par_dict, t=1., tp=0., dim=3, nk=100, S_iw_list=None, refine=True, nk_avg=(1,1), fit=False, n_iw_extr=True, save_file=None, verbose=True):
    """
    Run a 1D sweep of DMFT calculations over a list of tuples (T, U or n).
    """

    start_time = time.time()

    lattice = LATTICE(t=t, tp=tp, dim=dim)

    # Extract the parameter list for the loop
    list_label = next(k for k, v in par_dict.items() if isinstance(v, (list, np.ndarray)))

    if S_iw_list is None:
        S_iw_list = [None]*len(par_dict[list_label])
    
    # Sweep initialization
    results = {'t': lattice.t, 'dim': lattice.dim, 'nk': nk,
               'S_iw_list': S_iw_list, 'refine': refine, 'n_iw_extr': n_iw_extr,
               'invchi': [],  'Q': [], 'par_list': []}

    nk_list = [nk - i*nk_avg[1] for i in range(nk_avg[0])]
    Q_avg = np.zeros((len(par_dict[list_label]), 3))
    invchi0_avg = np.zeros(len(par_dict[list_label]))
    total_jobs = len(par_dict[list_label])*len(nk_list)
    for i_nk, nk_val in enumerate(nk_list):

        lattice.get_e_k(nk_val)
        if refine:
            lattice.get_phase_k(nk_val)

        q_path = np.array([[1.,1.,qz] for qz in np.linspace(0.,1.,nk_val//2+1)])
        
        # Parameter sweep
        for i_var, var in enumerate(par_dict[list_label]):

            par = {**par_dict, list_label: var}
            U, T, n = par['U'], par['T'], par['n']

            mu = get_mu(e_k=lattice.e_k, n_goal=n, beta=1/T)

            Q, invchi0_Q = get_min_invchi0(lattice, mu, beta=1/T, q_path=q_path, S_iw_arr=S_iw_list[i_var], refine=refine, n_iw_extr=n_iw_extr)
            
            r"""
                iw_mesh = MeshImFreq(beta=1/T, S='Fermion', n_iw=len(S_iw_list[i_var]))
                S_iw = Gf(mesh=iw_mesh, target_shape=[1,1])
                S_iw << 0

                G_iwk = lattice_dyson_g_wk(mu, lattice.e_k, S_iw)
                chi0 = imtime_bubble_chi0_wk(G_iwk, nw=1, verbose=False)
                chi0_mesh = 1/chi0.data[0, :, 0, 0]

                max_idx = np.unravel_index(np.argmin(chi0_mesh), chi0_mesh.shape)
                Q = 2*np.array(max_idx) / nk
                invchi0_Q = chi0_mesh[max_idx]
            """

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

    if fit:
        pos_mask = results['invchi'] > 0
        x_fit = np.array(par_dict[list_label])[pos_mask]
        y_fit = results['invchi'][pos_mask]
        if len(x_fit) >= 2:
            try:
                par, cov = curve_fit(critical, x_fit, y_fit, p0=[2., 1., x_fit[0]], maxfev=10000)
                
                results['ampl'] = par[0]
                results['gamma'] = par[1]
                results['Xc'] = par[2]

                results['ampl_err'] = cov[0,0]
                results['gamma_err'] = cov[1,1]
                results['Xc_err'] = cov[2,2]

                Q_vals = results['Q'][pos_mask][:2]
                mQ = (Q_vals[1] - Q_vals[0])/(x_fit[1] - x_fit[0])
                results['Qc'] = Q_vals[0] - mQ * (x_fit[0] - results['Xc']) 

            except RuntimeError as e:
                print(f"\nFit failed: {e}")
                
                for key in ['Xc', 'gamma', 'ampl', 'Xc_err', 'gamma_err', 'ampl_err']:
                    results[key] = 0.
                
                results['Qc'] = np.array([0.]*dim)
        else:
            print("\nFit skipped: fewer than 2 positive points")
            
            for key in ['Xc', 'gamma', 'ampl', 'Xc_err', 'gamma_err', 'ampl_err']:
                results[key] = 0.
                
            results['Qc'] = np.array([0.]*dim)

    if save_file is not None:
        HDFwrite_dict(save_file, results)
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\rCompleted {len(par_dict[list_label])} jobs in {elapsed_time:.2f} seconds | U={U:.3f}, T={T:.3f}, n={n:.3f}")
    
    return results