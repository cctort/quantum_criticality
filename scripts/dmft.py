import numpy as np
from scipy.optimize import fsolve
from triqs.gf import *
from triqs.plot.mpl_interface import *
from triqs.lattice.tight_binding import dos
from triqs.dos import HilbertTransform
from my_dmft.scripts.utils import *

class DMFT:
    """
    Class containing all DMFT-related objects and methods
    """
    def __init__(self, lattice, beta, n_iw=1024, nk=100, n_eps=100):
        # Input parameters
        self.lattice = lattice
        self.beta = beta
        self.n_iw = n_iw
        self.nk = nk
        self.n_eps = n_eps

        # Matsubara frequency
        self.iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=n_iw)
        self.G_iw = Gf(mesh=self.iw_mesh, target_shape=[1,1])
        self.G0_iw = self.G_iw.copy()
        self.S_iw = self.G_iw.copy()
        self.D_iw = self.G_iw.copy()
        self.S2_iw = self.G_iw.copy()
        self.G_temp = self.G_iw.copy()
        self.S_temp = self.G_iw.copy()
        
        # Imaginary time
        self.tau_mesh = MeshImTime(beta=beta, S='Fermion', n_tau=8*n_iw+1)
        self.G0_tau = Gf(mesh=self.tau_mesh, target_shape=[1,1])
        self.S2_tau = self.G0_tau.copy()
        
        # Non-interacting density of states
        rho = dos(self.lattice.H_r.tb, nk, n_eps, name='')[0] # First band

        # G(z) = \int rho(eps) / (z - eps) deps
        ht = HilbertTransform(rho)

        self.eps_vals = rho.eps
        self.rho_vals = ht.rho_for_sum


    def get_S2_iw(self, U):
        # Fourier transform G0 to imaginary time
        self.G0_tau << Fourier(self.G0_iw)
        
        # Second-order self-energy in time: Sigma(tau) = U^2 * G0(tau)^2 * G0(-tau)
        self.S2_tau << (U**2) * self.G0_tau * self.G0_tau * self.G0_tau.conjugate()
        
        # Fourier transform back to frequency
        self.S2_iw << Fourier(self.S2_tau)
    
    def full_IPT(self, S_iw, U, n, n0, mu, mu0):

        # IPT Ansatz: Sigma(iw) = U*n + Sigma_2(iw) / (1 - B*Sigma_2(iw))
        A = (n * (1. - n)) / (n0 * (1. - n0))
        B = ((1. - n) * U + (mu0 - mu)) / (n0 * (1. - n0) * U**2)

        S_iw << U * n + (A * self.S2_iw) * inverse(1. - B * self.S2_iw)

    def simple_IPT(self, S_iw, U, n):
        
        # Simple form (exact at half-filling)
        S_iw << U * n + self.S2_iw

    def square_gloc(self, G_iw, S_iw, mu):

        z_iw = np.array([complex(w) for w in self.iw_mesh]) + mu - S_iw.data[:,0,0]

        z = z_iw[:, np.newaxis] # Shape (n_iw, 1)
        e = self.eps_vals[np.newaxis, :] # Shape (1, n_eps)

        G_iw.data[:,0,0] = np.sum(self.rho_vals / (z - e), axis=1)

        # G_iw << self.ht(S_iw) # might cause a memory leak (.copy() inside source code)


    def solve_n0(self, mu0_val, n_goal):
        # n0 depends only on mu0 and Delta
        self.G_temp << inverse(iOmega_n + mu0_val[0] - self.D_iw)
    
        return self.G_temp.total_density().real - n_goal

    def solve_n(self, mu_val, n_goal, mu0_val, n0_val, U, solver):
        # n depends on mu, n0, mu0 and Sigma
        if solver == 'simple':
            self.simple_IPT(self.S_temp, U, n_goal)
        elif solver == 'full':
            self.full_IPT(self.S_temp, U, n_goal, n0_val, mu_val[0], mu0_val)

        self.square_gloc(self.G_temp, self.S_temp, mu_val[0])
    
        return self.G_temp.total_density().real - n_goal


    def run(self, U, n_goal=0.5, init_S=None, init_mu=None, init_label='metal', max_steps=1000, alpha=1., diff_tol=1e-10, slope_tol=1e-1, refinement=True, solver=None, half=False, verbose=True):

        # Initial n and n0
        n0, n = n_goal, n_goal
        
        if init_S is None:

            # Initial mu and mu0
            mu0 = 1e-5
            mu = U * n_goal
                
            if init_label == 'metal':
                
                # Initial self-energy
                self.S_iw << U * n_goal

            elif init_label == 'atom':

                # Initial self-energy
                denom = iOmega_n + mu - U * (1. - n_goal)
                self.S_iw << U * n_goal + n_goal * (1.-n_goal) * (U**2) * inverse(denom)
                
        else:

            # Initial self-energy
            self.S_iw = init_S.copy()

            # Initial mu and mu0
            if init_mu is None:
                mu, mu0 = U/2., 1e-5
            else:
                mu, mu0 = init_mu
        
        # If no IPT solver is specified, use general one for n != 0.5
        if solver is None:
            if n_goal == 0.5:
                solver = 'simple'
            else:
                solver = 'full'

        self.convergence = {'diff': [], 'alpha': [], 'n': [], 'mu': [], 'n0': [], 'mu0': []}
        
        # Initial guess for hybridization from Hartree state
        self.square_gloc(self.G_iw, self.S_iw, mu)
        self.D_iw << iOmega_n + mu - self.S_iw - inverse(self.G_iw)
        
        if verbose:
            print('='*50)
            print(f"Starting IPT DMFT: U={U:.3f}, T={1/self.beta:.3f}, n={n_goal:.3f}")
        
        steps = 0
        converged = False
        while steps < max_steps:

            S_iw_old = self.S_iw.data[:,0,0].copy()
            
            if half:
                mu0 = 0.
            else:
                # Get mu0 such that n0=n_goal
                try:
                    mu0 = fsolve(self.solve_n0, mu0, args=(n_goal,))[0]
                except Exception as e:
                    if verbose:
                        print(f'Error during newton root finder ({e})')
                    break


            self.G0_iw << inverse(iOmega_n + mu0 - self.D_iw)
            n0 = self.G0_iw.total_density().real

            # Get mu such that n=n_goal
            self.get_S2_iw(U)

            if half:
                mu = U/2
            else:
                try:
                    mu = fsolve(self.solve_n, mu, args=(n_goal, mu0, n0, U, solver))[0]
                except Exception as e:
                    if verbose:
                        print(f'Error during newton root finder ({e})')
                    break

            if solver == 'simple': # A=1 and B=0, exact at half-filling
                self.simple_IPT(self.S_iw, U, n_goal)
                
            elif solver == 'full': # general formula
                self.full_IPT(self.S_iw, U, n_goal, n0, mu, mu0)

            # Linear mixing of self-energy
            self.S_iw.data[:,0,0] = (1-alpha)*S_iw_old + alpha*self.S_iw.data[:,0,0]

            # Get G_loc
            self.square_gloc(self.G_iw, self.S_iw, mu)
            n = self.G_iw.total_density().real

            self.convergence['mu'].append(mu)
            self.convergence['mu0'].append(mu0)
            self.convergence['n'].append(n)
            self.convergence['n0'].append(n0)

            # Update hybridization for next dmft loop
            self.D_iw << iOmega_n + mu - self.S_iw - inverse(self.G_iw)

            # Convergence check
            S_iw_new = self.S_iw.data[:,0,0].copy()
            diff = np.max(np.abs(S_iw_old - S_iw_new))
            self.convergence['diff'].append(diff)
            self.convergence['alpha'].append(alpha)
            steps += 1
            if steps%10 == 0:
                if verbose:
                    print(f'\rstep {steps:2d}: diff = {diff:.3e} | mu = {mu:.4f} | mu0 = {mu0:.4f}', end='')
                if not converged:
                    # Convergence check
                    if np.mean(self.convergence['diff'][-10:]) < diff_tol:
                        if not refinement:
                            break
                        converged = True
                        alpha /= 2

                else:
                    # Refinement after convergence
                    log_diff = np.log(self.convergence['diff'][-10:])
                    slope = np.polyfit(range(len(log_diff)), log_diff, 1)[0]
                    if verbose:
                        print(f'\rstep {steps:2d}: slope = {slope:.3f} | mu = {mu:.4f} | mu0 = {mu0:.4f}', end='')
                    if np.abs(slope) < slope_tol:
                        break
            
            # Reduce mixing parameter after enough steps
            #if steps == max_steps//2:
            #    alpha /= 2
        
        if converged:
            if verbose:
                print(f'\rConverged at step {steps:2d} with diff = {diff:.3e} | slope = {slope:.3f}')
                print(f'n = {n:.3f} | n0 = {n0:.3f} | mu = {mu:.4f} | mu0 = {mu0:.4f}')
        else:
            if verbose:
                print(f'\nNot converged, skipping this point')
        
        self.steps = steps
        self.converged = converged
            
def sweep_dmft(params, lattice, init_label, alpha=1., n_iw=1024, nk=100, n_eps=100, verbose=False):
    """
    Run a 1D sweep of DMFT calculations over a list of tuples (T, U or n).
    """

    # params rewritten as a list of U,T,n dicts
    key_list = next(k for k, v in params.items() if isinstance(v, (list, np.ndarray)))
    par_dict = [{**params, key_list: v} for v in params[key_list]]
    
    # Sweep initialization
    dmft = DMFT(lattice, 1/par_dict[0]['T'], n_iw, nk, n_eps)
    prev_S, prev_mu = None, None
    prev_U, prev_T = par_dict[0]['U'], par_dict[0]['T']
    results = {}

    # Parameter sweep
    for par in par_dict:
        
        # We create a new DMFT object if T changes
        if par['T'] != prev_T:
            dmft = DMFT(lattice, 1/par['T'], n_iw, nk, n_eps)

        # We reset initial (mu, mu0) if U changes
        if par['U'] != prev_U:
            prev_mu = None

        # Run DMFT
        dmft.run(U=par['U'], n_goal=par['n'], init_S=prev_S, init_mu=prev_mu, init_label=init_label, alpha=alpha, verbose=verbose)

        # Store results if converged
        if dmft.converged:

            par_key = tuple(par.items())

            # Results for each U,T,n point
            results[par_key] = {
                'G_iw': dmft.G_iw.data[:,0,0].copy(),
                'S_iw': dmft.S_iw.data[:,0,0].copy(),
                'steps': dmft.steps
                }
            
            # Convergence info and mu, n params (only final values)
            for conv_key in dmft.convergence.keys():
                results[par_key][conv_key] = dmft.convergence[conv_key][-1]

            # Update inputs the next iteration
            prev_S = dmft.S_iw.copy()
            prev_mu = dmft.convergence['mu'][-1], dmft.convergence['mu0'][-1]
            prev_T = par['T']
            prev_U = par['U']
        
    results['inputs'] = {
        't': lattice.t,
        'dim': lattice.dim,
        'init_label': init_label,
        'alpha': alpha,
        'n_iw': n_iw,
        'nk': nk,
        'n_eps': n_eps
    }
    
    return results

r"""
def sweep_parallel(sweep_list, lattice, init_label, alpha=1., n_iw=1024, nk=100, n_eps=100, n_jobs=-1):

    def _worker(params, lattice_args, init_label, alpha, n_iw, nk, n_eps):
        lattice = LATTICE(**lattice_args)
        return sweep_dmft(params, lattice, init_label, alpha, n_iw, nk, n_eps)

    lattice_args = {'t': lattice.t, 'dim': lattice.dim}
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_worker)(params, lattice_args, init_label, alpha, n_iw, nk, n_eps)
        for params in sweep_list
    )
    results = {}
    for r in results_list:
        results.update(r)
    return results
"""