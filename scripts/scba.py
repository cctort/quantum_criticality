"""
scba_triqs.py

SCBA self-energy for nearest-neighbour hopping disorder,
implemented in the TRIQS paradigm, using the LATTICE class interface.
Sweep logic mirrors sweep_dmft from dmft.py.
"""

import numpy as np
from scripts.obs import *
from numba import njit

@njit
def scba_solver(niw, Nk, v, k_vecs, v2_R_vecs, v2_vals, G_iwR):

    S_iwk = np.zeros((niw, Nk), dtype=np.complex128)
    for k in range(Nk):
        for n in range(niw):
            for i in range(len(v2_R_vecs)):
                phase = np.exp(1j * np.dot(v2_R_vecs[i].astype(np.float64), k_vecs[k]))
                S_iwk[n, k] += v * v2_vals[i] * G_iwR[n, i] * phase
    
    return S_iwk


class SCBA:

    def __init__(self, lat, niw, nk, disorder='nnn'):

        self.lat = lat
        self.niw = niw
        self.nk = nk
        self.Nk = nk**lat.dim
        self.Nk_ibz = len(self.lat.ibz_w_k)

        self.get_v2R(disorder)

        self.S_iwk = np.zeros((niw, self.Nk_ibz), dtype=np.complex128)

    def get_v2R(self, disorder):

        dim = self.lat.dim
        R0 = np.zeros(dim, dtype=int)

        if disorder == 'onsite':
            V_support = [R0]
        elif disorder == 'nn':
            V_support = [np.array(R[:dim]) for R in self.lat.R_vecs_NN]
        elif disorder == 'nnn':
            V_support = [np.array(R[:dim]) for R in self.lat.R_vecs_NNN]
        else:
            raise ValueError(f"Unknown disorder type: {disorder}")

        V2_dict = {}
        for R1 in V_support:
            for R2 in V_support:
                R = tuple(R1 - R2)
                V2_dict[R] = V2_dict.get(R, 0) + 1

        self.v2_R_vecs = np.array(list(V2_dict.keys()), dtype=int)    # (n_R, dim)
        self.v2_vals   = np.array(list(V2_dict.values()), dtype=float)  # (n_R,)

    def run(self, v, beta, n_goal=1., max_iter=200, tol=1e-10, mix=0., init_S=None, init_mu=0., ibz=True, verbose=True):
        
        dim = self.lat.dim
        nk = self.nk
        niw = self.niw
        mu = init_mu
        v2 = v**2

        if verbose:
            print('=' * 50)
            print(f"SCBA: nk={nk}^{dim}, niw={niw}, T={beta:.2f}, mu={mu:.4f}, v={v:.3f}")

        self.S_iwk, _ = get_iwk_arr(init_S, self.Nk_ibz, niw)

        self.run_stats = {'diff': [], 'mix': [], 'n': [], 'mu': []}

        converged = False
        for step in range(max_iter):

            mu = get_mu(self.lat.e_k, n_goal, beta, niw, self.S_iwk, self.lat.ibz_w_k)

            G_iwk = get_G_iwk(mu, beta, self.lat.e_k, self.S_iwk, self.niw)

            if ibz:
                G_iwk = self.lat.unfold_f_iwk(G_iwk)
            G_iwR = self.lat.get_f_iwR(G_iwk, self.nk).reshape(niw,-1)

            strides = nk**np.arange(dim - 1, -1, -1)
            sparse_idx = (self.v2_R_vecs % nk) @ strides
            G_iwR_sparse = G_iwR[:, sparse_idx]
            
            S_new = scba_solver(niw, self.Nk_ibz, v2, self.lat.k_vecs, self.v2_R_vecs, self.v2_vals, G_iwR_sparse)

            diff = float(np.max(np.abs(S_new - self.S_iwk)))
            self.S_iwk = mix * self.S_iwk + (1 - mix) * S_new

            n = density_iwk(self.lat.e_k, self.S_iwk, mu, beta, self.lat.ibz_w_k)

            self.run_stats['diff'].append(diff)
            self.run_stats['mix'].append(mix)
            self.run_stats['n'].append(n)
            self.run_stats['mu'].append(mu)

            if step % 1 == 0 and verbose:
                print(f"\rstep {step:4d}: diff = {diff:.3e}, mu = {mu:.3e}", end='')

            if diff < tol:
                converged = True
                if verbose:
                    print(f"\rConverged at step {step} with diff = {diff:.3e}, mu = {mu:.3e}")
                break

        if not converged:
            if verbose:
                print(f"\nNot converged after {max_iter} iterations (diff = {diff:.3e})")
        
        self.run_stats['converged'] = converged


def sweep_scba(params, lattice, n_iw=512, verbose=False):
    
    key_list = next(k for k, v in params.items() if isinstance(v, (list, np.ndarray)))
    par_list = [{**params, key_list: v} for v in params[key_list]]

    scba       = SCBA(lattice, beta=1/par_list[0]['T'], n_iw=n_iw)
    prev_Sigma = None
    prev_T     = par_list[0]['T']
    prev_W     = par_list[0]['W']
    results    = {}

    for par in par_list:

        if par['T'] != prev_T:
            scba = SCBA(lattice, beta=1/par['T'], n_iw=n_iw)

        if par['W'] != prev_W:
            prev_Sigma = None

        scba.run(
            W          = par['W'],
            mu         = par['mu'],
            max_iter   = par.get('max_iter', 200),
            tol        = par.get('tol', 1e-7),
            mix        = par.get('mix', 0.4),
            init_Sigma = prev_Sigma,
            verbose    = verbose,
        )

        par_key = tuple(par.items())

        if scba.converged:
            results[par_key] = {
                'S_iwk' : scba.S_iwk.copy(),
                'G_iwk'     : scba.G_iwk.copy(),
                'G_loc'    : scba.G_loc.copy(),
                'filling'  : scba.filling(),
                'diff'     : scba.convergence['diff'][-1],
                'converged': True,
            }
            prev_Sigma = scba.S_iwk.copy()
        else:
            results[par_key] = {'converged': False}

        prev_T = par['T']
        prev_W = par['W']

    results['inputs'] = {'t': lattice.t, 'dim': lattice.dim, 'n_iw': n_iw}
    return results


def worker_scba(T, W, mu, lattice_args, n_iw=512, verbose=False, **kwargs):
    """Stateless worker for sweep_parallel. Rebuilds LATTICE internally."""
    from scripts.lattice import LATTICE
    lattice = LATTICE(**lattice_args)
    lattice.get_e_k(lattice_args['nk'])

    scba = SCBA(lattice, beta=1/T, n_iw=n_iw)
    scba.run(W=W, mu=mu, verbose=verbose, **kwargs)

    if scba.converged:
        return {
            'T': T, 'W': W, 'mu': mu,
            'S_iwk' : scba.S_iwk.copy(),
            'G_iwk'     : scba.G_iwk.copy(),
            'G_loc'    : scba.G_loc.copy(),
            'filling'  : scba.filling(),
            'diff'     : scba.convergence['diff'][-1],
            'converged': True,
        }
    return {'T': T, 'W': W, 'mu': mu, 'converged': False}