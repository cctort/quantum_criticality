import time
import numpy as np
from scripts.obs import get_mu, density_iwk, get_G_iw_slice
from scripts.utils import serialize, merge_results, HDFArchive


class SCBA:

    def __init__(self, lat, bz, niw):
        self.lat = lat
        self.bz  = bz
        self.niw = niw
        self.Nk_ibz = len(bz.w_k)

        self.S_iwk = np.zeros((niw, 1), dtype=np.complex128)
        self.run_stats = {}

    def run(self, par, v, init_S=None, max_iter=200, tol=1e-10, mix=0., init_mu=0.,
            store_inputs=True, verbose=True, file_name=None):

        # Stores every input inside a dictionary, same convention as run_rpa
        if store_inputs:
            run_data = locals().copy()
            run_data.pop('self')
            run_data = serialize(run_data)
        else:
            run_data = dict(par)

        T, n_goal = par['T'], par['n']
        beta = 1. / T
        v2 = v ** 2

        self._last_beta = beta
        self._last_v    = v
        self._last_n    = n_goal

        if init_S is None:
            self.S_iwk[:] = 0.
        else:
            self.S_iwk[:] = np.asarray(init_S, dtype=np.complex128).reshape(self.niw, 1)

        self.run_stats = {'diff': [], 'mix': [], 'n': [], 'mu': []}
        S_new = np.empty_like(self.S_iwk)

        mu = init_mu
        start_time = time.time()
        if verbose:
            print('=' * 50)
            print(f"SCBA (onsite): nk={self.bz.nk}^{self.lat.dim}, niw={self.niw}, "
                  f"T={T:.4f}, mu={mu:.4f}, v={v:.3f}")

        converged = False
        for step in range(max_iter):
            mu = get_mu(self.bz.e_k, n_goal, beta, self.niw, self.S_iwk, self.bz.w_k)

            for n in range(self.niw):
                # G(iw_n, k) = 1 / (iw_n + mu - e_k - Sigma(iw_n)), all k in IBZ.
                G_k_ibz = get_G_iw_slice(mu, beta, self.bz.e_k, self.S_iwk, n)
                # Sigma(iw_n) = v^2 * G_loc(iw_n) = v^2 * <G(iw_n,k)>_BZ.
                S_new[n, 0] = v2 * np.dot(self.bz.w_k, G_k_ibz)

            diff = float(np.abs(S_new - self.S_iwk).max())

            if mix > 0.:
                self.S_iwk *= mix
                self.S_iwk += (1. - mix) * S_new
            else:
                self.S_iwk[:] = S_new

            n_el = density_iwk(self.bz.e_k, self.S_iwk, mu, beta, self.bz.w_k)

            self.run_stats['diff'].append(diff)
            self.run_stats['mix'].append(mix)
            self.run_stats['n'].append(n_el)
            self.run_stats['mu'].append(mu)

            if verbose:
                print(f"\rstep {step:4d}: diff={diff:.3e}  mu={mu:.6f}", end='')

            if diff < tol:
                converged = True
                if verbose:
                    print(f"\rConverged at step {step:4d}  "
                          f"diff={diff:.3e}  mu={mu:.6f}")
                break

        if not converged and verbose:
            print(f"\nNot converged after {max_iter} steps  (diff={diff:.3e})")

        self.run_stats['converged'] = converged

        run_data['S_iwk']     = self.S_iwk.copy()
        run_data['mu']        = mu
        run_data['n']         = n_el
        run_data['diff']      = diff
        run_data['converged'] = converged
        run_data['run_stats'] = self.run_stats

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"\nv={v:.5g}, T={T:.5g}, n={n_goal:.5g}: completed after {elapsed_time:.1f}s  "
                  f"n={n_el:.5g}, mu={mu:.5g}, converged={converged}")

        if file_name is not None:
            with HDFArchive(file_name, 'w') as ar:
                for key, val in run_data.items():
                    ar[key] = val

        return run_data

    def sweep(self, par_list, v, init_S=None, max_iter=200, tol=1e-10, mix=0., init_mu=0.,
              carry_solution=True, verbose=True, file_name=None):

        # Stores every input inside a dictionary, same convention as sweep_rpa
        sweep_data = {k: v for k, v in locals().items() if k not in ('self', 'par_list')}

        if verbose:
            print('=' * 50)
            start_time = time.time()

        sweep_length = len(par_list)

        results_list = []
        for i, par in enumerate(par_list):
            # After the first point, carry the previous converged self-energy over
            # as the initial guess for the next parameter set (unless disabled) -
            # cheaper than restarting from S=0 at every point in the sweep.
            S0 = init_S if (i == 0 or not carry_solution) else self.S_iwk

            result = self.run(par, v, init_S=S0, max_iter=max_iter, tol=tol, mix=mix,
                               init_mu=init_mu, store_inputs=False, verbose=False,
                               file_name=None)
            results_list.append(result)

            if verbose:
                print(f"\r[{i + 1:4d}/{sweep_length}] "
                      f"v={v:.4g}, T={par['T']:.4g}, n={par['n']:.4g}: "
                      f"converged={result['converged']}, diff={result['diff']:.2e}, "
                      f"mu={result['mu']:.5g}", end='')

        merged = merge_results(results_list, ['S_iwk', 'mu', 'n', 'diff', 'converged'])
        sweep_data.update(merged)

        if file_name is not None:
            with HDFArchive(file_name, 'w') as ar:
                for key, val in sweep_data.items():
                    ar[key] = val

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"\nCompleted {sweep_length} runs in {elapsed_time:.1f} seconds")

        return serialize(sweep_data)