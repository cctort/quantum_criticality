import numpy as np
from scripts.obs import get_mu, density_iwk, get_G_iw_slice


class SCBA:

    def __init__(self, lat, bz, niw):
        self.lat = lat
        self.bz  = bz
        self.niw = niw
        self.Nk_ibz = len(bz.w_k)

        self.S_iwk = np.zeros((niw, 1), dtype=np.complex128)
        self.run_stats = {}

    def run(self, v, beta, n_goal=1., init_S=None, max_iter=200, tol=1e-10, mix=0., init_mu=0., verbose=True):
        
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
        if verbose:
            print('=' * 50)
            print(f"SCBA (onsite): nk={self.bz.nk}^{self.lat.dim}, niw={self.niw}, "
                  f"T={1/beta:.4f}, mu={mu:.4f}, v={v:.3f}")

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
        return self.S_iwk