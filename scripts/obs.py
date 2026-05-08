import numpy as np
from triqs.gf import GfImFreq, GfReFreq, MeshReFreq
from triqs_tprf.lattice_utils import imtime_bubble_chi0_wk
from triqs_tprf.lattice import lattice_dyson_g0_wk, lattice_dyson_g_wk, lindhard_chi00
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scripts.utils import *
from scripts.lattice import *
import time
from scipy.optimize import curve_fit
import numba as nb
from numba import njit

def get_A_iw0k(G_iwk, n_pade=60):

    N_iw = G_iwk.data.shape[0]
    N_k  = G_iwk.data.shape[1]

    beta = G_iwk.mesh.components[0].beta
    n_iw = N_iw // 2

    G_iw = GfImFreq(beta=beta, statistic='Fermion',
                    n_points=n_iw, target_shape=[1, 1])

    w_mesh = MeshReFreq(w_min=0.0, w_max=0.0, n_w=1)
    G_w = GfReFreq(mesh=w_mesh, target_shape=[1, 1])

    n_pade = min(n_pade, n_iw)

    A_k = np.empty(N_k)

    for ik in range(N_k):
        G_iw.data[:, 0, 0] = G_iwk.data[:, ik, 0, 0]

        G_w.set_from_pade(G_iw, n_points=n_pade)

        A_k[ik] = -G_w.data[0, 0, 0].imag / np.pi

    return A_k

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

@njit
def blochl_weights(e, mu):
    """
    Blöchl PRB 49, 16223 (1994) Appendix B.
    e: sorted corner energies (4,), e[0] <= e[1] <= e[2] <= e[3]
    Returns w[4]: contribution to integration weights (VT/VG factored out).
    sum(w) = fractional occupation of tetrahedron.
    """
    w = np.zeros(4)
    e1, e2, e3, e4 = e[0], e[1], e[2], e[3]

    if mu <= e1:
        return w

    if mu >= e4:
        w[0] = w[1] = w[2] = w[3] = 0.25
        return w

    # shorthand: eij = ei - ej  (Blöchl notation)
    e21 = e2 - e1;  e31 = e3 - e1;  e41 = e4 - e1
    e32 = e3 - e2;  e42 = e4 - e2;  e43 = e4 - e3

    # avoid exact degeneracies
    if abs(e21) < 1e-14: e21 = 1e-14
    if abs(e31) < 1e-14: e31 = 1e-14
    if abs(e41) < 1e-14: e41 = 1e-14
    if abs(e32) < 1e-14: e32 = 1e-14
    if abs(e42) < 1e-14: e42 = 1e-14
    if abs(e43) < 1e-14: e43 = 1e-14

    if mu <= e2:
        # Blöchl B2-B6: Case I,  C = (eF-e1)^3 / (4 * e21*e31*e41)
        C = (mu - e1)**3 / (4.0 * e21 * e31 * e41)
        w[0] = C * (4.0 - (mu-e1) * (1.0/e21 + 1.0/e31 + 1.0/e41))
        w[1] = C * (mu-e1) / e21
        w[2] = C * (mu-e1) / e31
        w[3] = C * (mu-e1) / e41

    elif mu <= e3:
        # Blöchl B7-B13: Case II
        C1 = (mu-e1)**2 / (4.0 * e41 * e31)
        C2 = (mu-e1) * (mu-e2) * (e3-mu) / (4.0 * e41 * e32 * e31)
        C3 = (mu-e2)**2 * (e4-mu) / (4.0 * e42 * e32 * e41)
        w[0] = C1 + (C1+C2)*(e3-mu)/e31 + (C1+C2+C3)*(e4-mu)/e41
        w[1] = C1+C2+C3 + (C2+C3)*(e3-mu)/e32 + C3*(e4-mu)/e42
        w[2] = (C1+C2)*(mu-e1)/e31 + (C2+C3)*(mu-e2)/e32
        w[3] = (C1+C2+C3)*(mu-e1)/e41 + C3*(mu-e2)/e42

    else:
        # Blöchl B14-B18: Case III,  C = (e4-eF)^3 / (4 * e41*e42*e43)
        C = (e4 - mu)**3 / (4.0 * e41 * e42 * e43)
        w[0] = 0.25 - C * (e4-mu) / e41
        w[1] = 0.25 - C * (e4-mu) / e42
        w[2] = 0.25 - C * (e4-mu) / e43
        w[3] = 0.25 - C * (4.0 - (e4-mu) * (1.0/e41 + 1.0/e42 + 1.0/e43))

    return w

def check_blochl_weights(e, mu):
    """
    Sum of weights must equal the integrated DOS fraction:
    - Case I:  (mu-e1)^3 / [6*(e2-e1)*(e3-e1)*(e4-e1)]
    - Case II: (C1+C2+C3)/6
    - Case III: 1/4 - (e4-mu)^3 / [6*(e4-e1)*(e4-e2)*(e4-e3)]
    """
    w = blochl_weights(e, mu)
    w_sum = np.sum(w)
    
    e1,e2,e3,e4 = e
    def safe(x): return x if abs(x)>1e-14 else 1e-14
    
    if mu <= e1:
        expected = 0.0
    elif mu <= e2:
        d = mu-e1
        expected = d**3 / (6*safe(e2-e1)*safe(e3-e1)*safe(e4-e1))
    elif mu <= e3:
        d1=mu-e1; d2=mu-e2
        C1 = d1**2 / (safe(e4-e1)*safe(e3-e1))
        C2 = d1*d2  / (safe(e4-e1)*safe(e3-e2))
        C3 = d2**2  / (safe(e4-e2)*safe(e3-e2))
        expected = (C1+C2+C3)/6
    elif mu <= e4:
        d = e4-mu
        expected = 0.25 - d**3/(6*safe(e4-e1)*safe(e4-e2)*safe(e4-e3))
    else:
        expected = 1.0
    
    return w_sum, expected, abs(w_sum - expected)

@njit
def lindhard_blochl(mu, beta, e_k, e_kq, tetrahedra, S_k=0., S_kq=0.):
    chi0 = 0.0

    for tet in tetrahedra:
        ek_tet  = e_k[tet]  + S_k[tet].real
        ekq_tet = e_kq[tet] + S_kq[tet].real

        order = np.argsort(ek_tet)
        ek_s   = ek_tet[order]
        ekq_s  = ekq_tet[order]

        wk = blochl_weights(ek_s, mu)

        for i in range(4):
            fk  = 1.0 if ek_s[i] < mu else 0.0
            fkq = 1.0 if ekq_s[i] < mu else 0.0
            denom = ek_s[i] - ekq_s[i]

            if abs(denom) > 1e-9:
                chi0 += wk[i] * (fk - fkq) / denom
            # denom ~ 0 and fk=fkq → contribution is 0, skip

    return chi0 / len(tetrahedra)

@njit
def cexp(z):
    zr = z.real
    zi = z.imag
    ezr = np.exp(zr)
    return ezr * np.cos(zi) + 1j * ezr * np.sin(zi)

@njit
def lindhard2(mu, beta, e_k, e_kq, S_k, S_kq):

    Nk = len(e_k)
    chi0_sum = 0.0

    for k in range(Nk):
        
        e_k_eff = e_k[k] + S_k[k].real
        e_kq_eff = e_kq[k] + S_kq[k].real
        Gamma_k = - S_k[k].imag
        Gamma_kq = - S_kq[k].imag

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
        elif xkq.real < -500.0:
            nF_kq = 1.0 + 0.0j
        else:
            nF_kq = 1.0 / (cexp(xkq) + 1.0)

        de = (e_k_eff - e_kq_eff) - 1j*(Gamma_k + Gamma_kq)

        if abs(de) < 1e-8:
            chi0 = - beta * nF_k * (1.0 - nF_k)
        else:
            chi0 = - (nF_k - nF_kq) / de

        chi0_sum += chi0

    return chi0_sum.real / Nk

@njit
def lindhard(mu, beta, e_k, e_kq, S_k, S_kq):

    Nk = len(e_k)
    chi0_sum = 0.0

    for k in range(Nk):
        
        e_k_eff = e_k[k] + S_k[k].real
        e_kq_eff = e_kq[k] + S_kq[k].real
        Gamma_k = - S_k[k].imag
        Gamma_kq = - S_kq[k].imag

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
        elif xkq.real < -500.0:
            nF_kq = 1.0 + 0.0j
        else:
            nF_kq = 1.0 / (cexp(xkq) + 1.0)

        de = (e_k_eff - e_kq_eff) - 1j*(Gamma_k - Gamma_kq)

        if abs(de) < 1e-8:
            chi0 = - beta * nF_k * (1.0 - nF_k)
        else:
            chi0 = - (nF_k - nF_kq) / de

        chi0_sum += chi0

    return chi0_sum.real / Nk

@njit
def matsubara_sum(mu, beta, e_k, e_kq, S_iwk, S_iwkq):
    Nk = len(e_k)
    chi0 = 0.0 + 0.0j
    for n in range(len(S_iwk)):
        iw = 1j * (2*n + 1) * np.pi / beta
        for k in range(Nk):
            G_iwk  = 1.0 / (iw + mu - e_k[k] - S_iwk[n,k])
            G_iwkq = 1.0 / (iw + mu - e_kq[k] - S_iwkq[n,k])
            chi0 += G_iwk * G_iwkq
    return (-2.0 / beta * chi0).real / Nk

def get_iwk_arr(S_val, Nk, dim, n_iw=1):
    k_dep = False

    if S_val is None or isinstance(S_val, (int, float, complex)):
        val = 0j if S_val is None else complex(S_val)
        # if caller wants n_iw > 1 with a constant sigma, broadcast over frequencies too
        S_val = np.full((n_iw, 1), val, dtype=complex)

    else:
        S_val = np.asanyarray(S_val, dtype=complex)

        if S_val.ndim == 1:
            if len(S_val) == Nk:
                # k-dependent, no frequency axis — replicate to n_iw if requested
                S_val = np.tile(S_val[np.newaxis, :], (n_iw, 1))  # (n_iw, Nk)
                k_dep = True
            else:
                # k-independent, frequency-dependent: (len,) → (len, 1)
                # input length takes priority over n_iw
                n_iw = len(S_val)
                S_val = S_val[:, np.newaxis]                        # (n_iw, 1)

        elif S_val.ndim == dim:
            # k-dependent spatial array, no frequency axis
            S_val = np.tile(S_val.reshape(1, Nk), (n_iw, 1))      # (n_iw, Nk)
            k_dep = True

        else:
            # (n_iw', Nk) already — n_iw' takes priority
            n_iw = S_val.shape[0]
            k_dep = True

    return S_val, n_iw, k_dep

def compute_chi0_fft(lattice, mu, T, S_iwk, n_iw, nk, n_iw_extr=False):

    beta = 1.0 / T
    e_k  = lattice.e_k.data[:, 0, 0].real.reshape(nk, nk, nk)
    has_sigma = not np.allclose(S_iwk, 0)

    def _run(n_iw_run):
        iw_n  = (2*np.arange(n_iw_run) + 1) * np.pi / beta
        chi0_r = np.zeros((nk, nk, nk), dtype=np.complex128)
        G_r    = np.zeros((nk, nk, nk), dtype=np.complex128)

        for n in range(n_iw_run):
            if has_sigma:
                sig = S_iwk[n, 0] if S_iwk.shape[1] == 1 else S_iwk[n].reshape(nk, nk, nk)
            else:
                sig = 0.0

            G_k = 1.0 / (1j*iw_n[n] + mu - e_k - sig)
            np.fft.ifftn(G_k, out=G_r)
            del G_k

            chi0_r.real += G_r.real * G_r.real - G_r.imag * G_r.imag
            chi0_r.imag += 2.0 * G_r.real * G_r.imag

        chi0_q = np.fft.fftn(chi0_r)
        chi0_q *= -2.0 / beta
        invchi0_mesh = chi0_q.real.copy()
        np.reciprocal(invchi0_mesh, out=invchi0_mesh)
        return invchi0_mesh

    invchi0_mesh = _run(n_iw)
    min_idx   = np.unravel_index(np.argmin(invchi0_mesh), invchi0_mesh.shape)
    Q         = np.sort(1 - np.abs(2*np.array(min_idx)/nk - 1))[::-1]
    invchi0_Q = invchi0_mesh[min_idx]

    if n_iw_extr:
        # same subsampling scheme as get_min_invchi0
        n_iw_list      = [n_iw]
        invchi0_Q_list = [invchi0_Q]

        for s in [2, 3, 4]:
            n_sub = n_iw // s
            if n_sub < 4:
                break
            mesh_sub  = _run(n_sub)
            invchi0_Q_list.append(mesh_sub[min_idx])  # same q-point as full run
            n_iw_list.append(n_sub)

        coeff     = np.polyfit(1./np.array(n_iw_list), invchi0_Q_list, 2)
        invchi0_Q = coeff[-1]  # extrapolated value at 1/n_iw → 0

    return Q, invchi0_Q

def get_min_invchi0(lattice, mu, beta, q_path=None, method='lindhard', n_iw=1, S_iwk=None, refine=True, n_iw_extr=False):

    nk = lattice.nk
    dim = lattice.dim

    S_iwk, n_iw, k_dep = get_iwk_arr(S_iwk, nk**dim, dim, n_iw)

    def expand(S, Nk):
        """Tile only if k-independent, and only to the path length Nk."""
        return S if S.shape[1] > 1 else np.tile(S, (1, Nk))

    invchi_grid = []
    for q in q_path:
        e_k, e_kq = lattice.get_e_kq(q, 'coarse')

        if k_dep:
            S_iwkq = lattice.get_f_iwkq(S_iwk, q, 'coarse')
        else:
            S_iwkq = S_iwk

        S_k  = expand(S_iwk,  len(e_k))
        S_kq = expand(S_iwkq, len(e_kq))

        if method == 'lindhard':
            invchi_grid.append(1/lindhard(mu, beta, e_k, e_kq, S_k[0], S_kq[0]))
        elif method == 'lindhard2':
            invchi_grid.append(1/lindhard2(mu, beta, e_k, e_kq, S_k[0], S_kq[0]))
        elif method == 'matsubara':
            invchi_grid.append(1/matsubara_sum(mu, beta, e_k, e_kq, S_k, S_kq))

    idx = np.argmin(invchi_grid)
    best_q = q_path[idx]
    invchi0_min = invchi_grid[idx]

    if refine and q_path is not None:

        start = q_path[0]
        end   = q_path[-1]
        step  = 1 / (len(q_path) - 1)

        if k_dep:
            S_iwR = lattice.get_f_iwR(S_iwk)

        def invchi0(s, S_iwk_arg):

            q = start + s * (end - start)
            e_k, e_kq = lattice.get_e_kq(q, 'fine')

            if k_dep:
                S_iwkq = lattice.get_f_iwkq(S_iwk_arg, q, 'fine', S_iwR)
            else:
                S_iwkq = S_iwk_arg

            S_k  = expand(S_iwk_arg, len(e_k))
            S_kq = expand(S_iwkq,    len(e_kq))

            if method == 'lindhard':
                return 1/lindhard(mu, beta, e_k, e_kq, S_k[0], S_kq[0])
            elif method == 'lindhard2':
                return 1/lindhard2(mu, beta, e_k, e_kq, S_k[0], S_kq[0])
            elif method == 'matsubara':
                return 1/matsubara_sum(mu, beta, e_k, e_kq, S_k, S_kq)

        s0 = idx / (len(q_path) - 1)
        res = minimize_scalar(invchi0, method='bounded',
                              bounds=(max(0, s0 - 5*step), min(1, s0 + 5*step)),
                              args=(S_iwk,))

        s_min = res.x
        best_q = start + s_min * (end - start)
        invchi0_min = invchi0(s_min, S_iwk)
    
    if n_iw_extr and method != 'lindhard':
        
        n_iw_list = [n_iw]
        invchi0_min_list = [invchi0_min]

        subs = [2, 3, 4]

        for s in subs:

            n_sub = n_iw//s
            n_iw_list.append(n_sub)
            invchi0_min_list.append(invchi0(s_min, S_iwk[:n_sub//2]))

        coeff = np.polyfit(1./np.array(n_iw_list), invchi0_min_list, 2)
        invchi0_min = coeff[-1]

    return best_q, invchi0_min

@njit
def _density_iw(eps, S_iwk, mu, beta):
    n_iw, Nk = S_iwk.shape
    n_tot = 0.0

    for k in range(Nk):
        ek = eps[k]
        xi_k = ek - mu

        g_sum = 0.0
        g0_sum = 0.0

        for n in range(n_iw):
            w = (2*n + 1) * np.pi / beta
            Sigma = S_iwk[n, k]

            denom_r = -ek + mu - Sigma.real
            denom_i = w - Sigma.imag
            g_sum += denom_r / (denom_r*denom_r + denom_i*denom_i)

            g0_sum += (-xi_k) / (w*w + xi_k*xi_k)

        # tail correction
        x = beta * xi_k
        if x > 500.0:
            nF = 0.0
        elif x < -500.0:
            nF = 1.0
        else:
            nF = 1.0 / (np.exp(x) + 1.0)

        tail = beta/2.0 * nF - 0.5

        g_corr = g_sum - g0_sum + tail
        n_tot += 2.0/beta * 2.0 * g_corr + 1.0

    return n_tot / Nk


def _density_static(eps, Phi, Gamma, mu, beta):
    x  = eps + Phi - mu
    xr = np.clip(beta * x, -500, 500)
    xi = beta * Gamma

    cos_xi = np.cos(xi)

    ex  = np.exp(xr)
    emx = np.exp(-xr)

    # stable form (no exp^2)
    nF_real = (cos_xi + emx) / (ex + 2*cos_xi + emx)

    return np.mean(2.0 * nF_real)


def get_mu(e_k, n_goal, beta, n_iw=1, S_iwk=None, dim=None):

    eps = e_k.data[:, 0, 0].real.flatten()
    S_iwk, n_iw, _ = get_iwk_arr(S_iwk, Nk=len(eps), dim=dim, n_iw=n_iw)

    # detect frequency independence
    freq_indep = np.allclose(S_iwk, S_iwk[0:1, :])

    if freq_indep:
        Sigma = S_iwk[0]
        Phi   = Sigma.real
        Gamma = Sigma.imag

        def density(mu):
            return _density_static(eps, Phi, Gamma, mu, beta)

    else:
        def density(mu):
            return _density_iw(eps, S_iwk, mu, beta)

    # root function
    def f(mu):
        return density(mu) - n_goal

    a = eps.min() - 10.0
    b = eps.max() + 10.0

    fa = f(a)
    fb = f(b)

    step = 10.0

    for _ in range(20):
        if fa * fb < 0:
            return brentq(f, a, b)

        a -= step
        b += step
        fa = f(a)
        fb = f(b)

    # fallback (avoid crashing parallel jobs)
    print("WARNING: could not bracket mu")
    print("f(a) =", fa, "f(b) =", fb)
    return np.nan

          
def sweep_chirpa(par_dict, t=1., tp=0., dim=3, nk=100, n_iw=1, S_iwk_list=None, method='lindhard', refine=True, nk_avg=(1,1), fit=False, fit_type=critical1, n_iw_extr=False, save_file=None, verbose=True):
    """
    Run a 1D sweep of DMFT calculations over a list of tuples (T, U or n).
    """

    start_time = time.time()

    lattice = LATTICE(t=t, tp=tp, dim=dim)

    # Extract the parameter list for the loop
    list_label = next(k for k, v in par_dict.items() if isinstance(v, (list, np.ndarray)))

    if isinstance(S_iwk_list, (list, np.ndarray)) and len(S_iwk_list) == len(par_dict[list_label]):
        pass
    else:
        S_iwk_list = [S_iwk_list] * len(par_dict[list_label])
    
    # Sweep initialization
    results = {'t': lattice.t, 'dim': lattice.dim, 'nk': nk, 'n_iw': n_iw,
               'S_iwk_list': S_iwk_list, 'refine': refine, 'n_iw_extr': n_iw_extr,
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

            mu = get_mu(e_k=lattice.e_k, n_goal=n, beta=1/T, S_iwk=S_iwk_list[i_var])
            
            if method == 'matsubara_tprf':

                S_iwk, n_iw, _ = get_iwk_arr(S_iwk_list[i_var], nk**dim, dim, n_iw=n_iw)
                iw_mesh = MeshImFreq(beta=1/T, S='Fermion', n_iw=n_iw)
                k_mesh = lattice.e_k.mesh
                iwk_mesh = MeshProduct(iw_mesh, k_mesh)
                S_iwk_gf = Gf(mesh=iwk_mesh, target_shape=[1,1])
                S_iwk_gf.data[:n_iw,:,0,0] = S_iwk[::-1].conj()
                S_iwk_gf.data[n_iw:,:,0,0] = S_iwk

                G_iwk = lattice_dyson_g_wk(mu, lattice.e_k, S_iwk_gf)
                chi0 = imtime_bubble_chi0_wk(G_iwk, nw=1, verbose=False, save_memory=True)
                invchi0_mesh = 1/chi0.data[0, :, 0, 0].reshape(nk,nk,nk)

                min_idx = np.unravel_index(np.argmin(invchi0_mesh), invchi0_mesh.shape)
                Q = np.sort(1-np.abs(2*np.array(min_idx) / nk-1))[::-1]
                invchi0_Q = invchi0_mesh[min_idx]

            elif method == 'matsubara_fft':

                S_iwk, n_iw, _ = get_iwk_arr(S_iwk_list[i_var], nk**dim, dim, n_iw=n_iw)
                Q, invchi0_Q = compute_chi0_fft(lattice, mu, T, S_iwk, n_iw, nk, n_iw_extr=n_iw_extr)

            else:

                Q, invchi0_Q = get_min_invchi0(lattice, mu, beta=1/T, q_path=q_path, S_iwk=S_iwk_list[i_var], method=method, refine=refine, n_iw_extr=n_iw_extr)

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
        x_fit = np.array(par_dict[list_label])[pos_mask][:15]
        y_fit = results['invchi'][pos_mask][:15]
        if len(x_fit) >= 2:
            try:
                
                par, cov = curve_fit(
                    fit_type,
                    x_fit,
                    y_fit,
                    p0=[1., 0.8, np.min(x_fit)*0.9],
                    bounds = ([0., 0., -np.inf], [np.inf, np.inf, np.inf]),
                    maxfev=10000
                )
                
                ampl, gamma, Xc = par

                results['ampl'] = ampl
                results['gamma'] = gamma
                results['Xc'] = Xc

                results['ampl_err'] = np.sqrt(cov[0,0])
                results['gamma_err'] = np.sqrt(cov[1,1])
                results['Xc_err'] = np.sqrt(cov[2,2])

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