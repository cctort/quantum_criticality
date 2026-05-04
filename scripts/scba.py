"""
scba_triqs.py

SCBA self-energy for nearest-neighbour hopping disorder,
implemented in the TRIQS paradigm, using the LATTICE class interface.
Sweep logic mirrors sweep_dmft from dmft.py.
"""

import numpy as np
from triqs.gf import Gf, MeshImFreq, MeshProduct, make_gf_from_fourier
from triqs_tprf.lattice import fourier_wk_to_wr, fourier_wr_to_wk, lattice_dyson_g_wk
from scipy.optimize import fsolve

class SCBA:

    def __init__(self, lattice, beta, xi=0.2, n_iw = 512, nk=50):

        self.lattice  = lattice
        self.beta     = beta
        self.n_iw     = n_iw
        self.dim        = lattice.dim
        self.nk       = nk

        # (iw, k) mesh
        self.k_mesh   = lattice.e_k.mesh
        self.iw_mesh  = MeshImFreq(beta=beta, S='Fermion', n_iw=n_iw)
        self.wk_mesh  = MeshProduct(self.iw_mesh, self.k_mesh)

        # Scattering matrix
        self.xi = xi
        self.tmat_r   = self.get_tmat_r()

        # Gf on (iw, k) mesh
        self.S_iwk = Gf(mesh=self.wk_mesh, target_shape=[1, 1])
        self.G_iwk = Gf(mesh=self.wk_mesh, target_shape=[1, 1])

        # Gf on iw mesh
        self.G_loc = Gf(mesh=self.iw_mesh, target_shape=[1, 1])

    def get_tmat_r(self):
        
        tmat_k = Gf(mesh=self.k_mesh, target_shape=[1, 1])
        k_arr  = np.array([k.value for k in self.k_mesh])[:, :self.dim]
        #tmat_k.data[:,0,0] = (np.cos(k_arr).mean(axis=-1)) ** 2
        k_arr = (k_arr + np.pi) % (2 * np.pi) - np.pi
        tmat_k.data[:,0,0] = np.exp(-(self.xi*np.pi)**2 * (k_arr**2).sum(axis=-1))
        return make_gf_from_fourier(tmat_k)
    
    def solve_n(self, mu_val, n_goal):

        G_iwk = lattice_dyson_g_wk(mu=-mu_val[0], e_k=self.lattice.e_k, sigma_wk=self.S_iwk)
        self.G_loc.data[:,0,0] = G_iwk.data[:,:,0,0].mean(axis=1)

        return 2*self.G_loc.total_density().real - n_goal


    def run(self, v, n_goal = 1., max_iter= 200,
            tol = 1e-10, mix = 1.,
            init_Sigma=None, constant=False, verbose = True):
        
        if init_Sigma is not None:
            self.S_iwk << init_Sigma
        else:
            self.S_iwk.zero()

        mu = 1e-5
        v **= 2

        self.convergence = {'diff': [], 'mix': [], 'n': [], 'mu': []}
        S_new = self.S_iwk.copy()

        if verbose:
            print('=' * 50)
            print(f"SCBA: nk={self.nk}^{self.dim}, beta={self.beta:.2f}, "
                  f"mu={mu:.4f}, scatt. pot.={np.sqrt(v):.3f}")

        for step in range(max_iter):

            mu = fsolve(self.solve_n, mu, args=(n_goal,))[0]

            self.G_iwk = lattice_dyson_g_wk(mu=mu, e_k=self.lattice.e_k, sigma_wk=self.S_iwk)

            if constant:
                S_new.data[:] = 0.0
                S_new.data[:,:,0,0] = v * self.G_iwk.data.mean(axis=1)[:,0,0][:, None]
            else:
                G_wr = fourier_wk_to_wr(self.G_iwk)
                G_wr.data[...] *= self.tmat_r.data[np.newaxis, :, ...]
                S_new << v * fourier_wr_to_wk(G_wr)

            diff = float(np.max(np.abs(S_new.data - self.S_iwk.data)))
            self.S_iwk << (1 - mix) * self.S_iwk + mix * S_new

            self.G_loc.data[:,0,0] = self.G_iwk.data[:,:,0,0].mean(axis=1)
            n = 2*self.G_loc.total_density().real

            self.convergence['diff'].append(diff)
            self.convergence['mix'].append(mix)
            self.convergence['n'].append(n)
            self.convergence['mu'].append(mu)

            if step % 1 == 0 and verbose:
                print(f"\rstep {step:4d}: diff = {diff:.3e}", end='')

            if diff < tol:
                self.converged = True
                if verbose:
                    print(f"\rConverged at step {step} with diff = {diff:.3e}")
                break
        else:
            if verbose:
                print(f"\nNot converged after {max_iter} iterations (diff = {diff:.3e})")

        self.G_iwk = lattice_dyson_g_wk(mu=mu, e_k=self.lattice.e_k, sigma_wk=self.S_iwk)
        self.G_loc.data[:,0,0] = self.G_iwk.data[:,:,0,0].mean(axis=1)


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