import numpy as np
from triqs.gf import *
from triqs.plot.mpl_interface import *
from triqs_tprf.tight_binding import TBLattice
from triqs.lattice import BravaisLattice
from scripts.utils import *  
import spglib

class LATTICE:
    """
    Class containing lattice informations
    """
    def __init__(self, t=1., tp=0., dim=3):

        self.t = t
        self.tp = tp
        self.dim = dim

        units = []
        hoppings = {}

        # nearest neighbors
        for i in range(dim):
            ax  = tuple(int(j==i) for j in range(dim))
            nax = tuple(-x for x in ax)
            units.append(ax)
            hoppings[ax]  = [[-t]]
            hoppings[nax] = [[-t]]

        # next-nearest neighbors (only meaningful in 2D or 3D)
        if dim >= 2 and tp != 0:
            nn_vectors = []
            if dim == 2:
                nn_vectors = [(1,1), (1,-1), (-1,1), (-1,-1)]
            elif dim == 3:
                # all 12 next-nearest neighbors in cubic lattice
                nn_vectors = [
                    (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
                    (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
                    (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)
                ]
            for vec in nn_vectors:
                hoppings[vec] = [[-tp]]

        self.a_vecs = np.array(units).T
        self.b_vecs = 2*np.pi * np.linalg.inv(self.a_vecs).T

        self.H_r = TBLattice(units=units, hoppings=hoppings)

        self.t_vals = np.array([t[0,0] for t in self.H_r.hoppings.values()])
        self.R_vecs = np.array([np.array(R[:dim]) for R in self.H_r.hoppings])
        
        unit_mat = np.eye(3)
        unit_mat[:dim, :dim] = np.array(units, dtype=float)
        self._spg_cell = (unit_mat, [[0., 0., 0.]], [1])

        self.k_vecs, self.k_vecs_fine = None, None
        self.e_k, self.e_k_fine = None, None

        self.ibz_w_k, self.ibz_w_k = None, None
        self.ibz_idx, self.ibz_idx_fine = None, None
        self.ibz_pos, self.ibz_pos_fine = None, None

    def get_bz(self, nk, ibz=False, fine=False):

        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)

        if not ibz:
            k_mesh = self.H_r.get_kmesh(nk_tuple)
            k_vecs = np.array([k.value for k in k_mesh])
            ibz_w_k = np.ones(nk**self.dim)

            if not fine:
                self.k_vecs, self.ibz_w_k = k_vecs, ibz_w_k
            else:
                self.k_vecs_fine, self.ibz_w_k_fine = k_vecs, ibz_w_k

        else:
            mapping, ir_grid = spglib.get_ir_reciprocal_mesh(nk_tuple, self._spg_cell)

            ibz_idx = np.unique(mapping)
            ibz_w_k = np.bincount(mapping)[ibz_idx].astype(float)
            ibz_pos  = np.searchsorted(ibz_idx, mapping)
            k_vecs = ir_grid[ibz_idx] / np.array(nk_tuple, dtype=float) * 2*np.pi

            if not fine:
                self.k_vecs, self.ibz_w_k = k_vecs, ibz_w_k
                self.ibz_idx, self.ibz_pos = ibz_idx, ibz_pos
            else:
                self.k_vecs_fine, self.ibz_w_k_fine = k_vecs, ibz_w_k
                self.ibz_idx_fine, self.ibz_pos_fine = ibz_idx, ibz_pos
    
    def get_e_k(self, fine=False):

        if not fine:
            H_k = self.H_r.fourier(self.k_vecs/(2*np.pi))
            self.e_k = H_k[:,0,0].real
        else:
            H_k = self.H_r.fourier(self.k_vecs_fine/(2*np.pi))
            self.e_k_fine = H_k[:,0,0].real

    def get_f_iwR(self, f_iwk, nk):
        dim = self.dim
        niw = f_iwk.shape[0]
        f_iwk_grid = f_iwk.reshape((niw,) + (nk,) * dim)

        f_iwR = np.fft.fftn(f_iwk_grid, axes=range(1, dim + 1)) / nk**dim

        return f_iwR

    def get_f_iwkq(self, f_iwk, q, nk, method='roll', f_iwR=None):
        
        dim = self.dim
        q = np.array(q)
        niw = f_iwk.shape[0]
        Nk = f_iwk.shape[1]

        if method == 'roll':
            f_iwk_grid = f_iwk.reshape((niw,) + (nk,) * dim)
            shifts = tuple(-round(qi * nk / 2) % nk for qi in q)
            f_iwkq = (np.roll(f_iwk_grid, shifts, axis=tuple(range(1, f_iwk_grid.ndim)))).reshape(f_iwk.shape)
            return f_iwkq

        elif method == 'exact':
            grid_1d = np.arange(nk)
            phase_grid = np.ones((nk,) * dim, dtype=complex)
            for alpha in range(dim):
                shape = [1] * dim
                shape[alpha] = nk
                phase_grid *= np.exp(1j * np.pi * q[alpha] * grid_1d.reshape(shape))

            f_iwkq = np.fft.ifftn(f_iwR * phase_grid[None, ...], axes=range(1, dim + 1)).reshape(f_iwk.shape) * Nk

            df_iwkq_dq = np.zeros((dim,) + f_iwk.shape)   # (dim, niw, nk^dim)
            for alpha in range(dim):
                shape = [1] * dim
                shape[alpha] = nk
                d_phase_grid = (1j * np.pi * grid_1d.reshape(shape)) * phase_grid
                df_iwkq_dq[alpha] = np.fft.ifftn(f_iwR * d_phase_grid[None, ...], axes=range(1, dim + 1)).reshape(f_iwk.shape) * Nk

            return f_iwkq, df_iwkq_dq

    def get_e_kq(self, e_k, q, nk, method='roll'):
        dim = self.dim
        q = np.array(q)
        Nk = len(e_k)

        if method == 'roll':
            e_k_grid = e_k.reshape((nk,) * dim)
            shifts = tuple(-round(qi * nk / 2) % nk for qi in q)
            e_kq = np.roll(e_k_grid, shifts, axis=tuple(range(dim))).reshape(-1)
            return e_kq

        elif method == 'exact':
            phase_q = np.exp(1j * np.pi * (self.R_vecs @ q))
            t_phase_q = self.t_vals * phase_q

            R_grid = np.zeros((nk,) * dim, dtype=complex)
            for tq, R in zip(t_phase_q, self.R_vecs):
                R_grid[tuple(int(r) % nk for r in R)] += tq
            e_kq = np.fft.ifftn(R_grid).real.reshape(-1) * Nk

            de_kq_dq = np.zeros((dim, Nk))
            for alpha in range(dim):
                dR_grid = np.zeros((nk,) * dim, dtype=complex)
                for tq, R in zip(1j * np.pi * self.R_vecs[:, alpha] * t_phase_q, self.R_vecs):
                    dR_grid[tuple(int(r) % nk for r in R)] += tq
                de_kq_dq[alpha] = np.fft.ifftn(dR_grid).real.reshape(-1) * Nk

            return e_kq, de_kq_dq

    def unfold_f_iwk(self, f_iwk_ibz, fine=False):
        ibz_pos = self.ibz_pos_fine if fine else self.ibz_pos
        return f_iwk_ibz[:, ibz_pos]

    def fold_f_iwk(self, f_iwk_full, fine=False):
        ibz_idx = self.ibz_idx_fine if fine else self.ibz_idx
        return f_iwk_full[:, ibz_idx]
    
    def unfold_f_k(self, f_k_ibz, fine=False):
        ibz_pos = self.ibz_pos_fine if fine else self.ibz_pos
        return f_k_ibz[ibz_pos]

    def fold_f_k(self, f_k_full, fine=False):
        ibz_idx = self.ibz_idx_fine if fine else self.ibz_idx
        return f_k_full[ibz_idx]