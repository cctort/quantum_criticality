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

        self.H_r = TBLattice(units=units, hoppings=hoppings)

        self.t_vals = np.array([t[0,0] for t in self.H_r.hoppings.values()])
        self.R_vecs = np.array([np.array(R[:dim]) for R in self.H_r.hoppings])
        
        unit_mat = np.eye(3)
        unit_mat[:dim, :dim] = np.array(units, dtype=float)
        self._spg_cell = (unit_mat, [[0., 0., 0.]], [1])

    def get_bz(self, nk, ibz=False, fine=False):

        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)

        k_mesh = self.H_r.get_kmesh(nk_tuple)
        full_k_vecs = np.array([k.value for k in k_mesh])
        ibz_w_k = np.ones(nk**self.dim)

        if not fine:
            self.k_vecs = full_k_vecs
            self.full_k_vecs = full_k_vecs
            self.ibz_w_k = ibz_w_k
        else:
            self.k_vecs_fine = full_k_vecs
            self.full_k_vecs_fine = full_k_vecs
            self.ibz_w_k_fine = ibz_w_k

        if ibz:
            mapping, ir_grid = spglib.get_ir_reciprocal_mesh(nk_tuple, self._spg_cell, is_shift=1)
            ir_idx = np.unique(mapping)
            ibz_w_k = np.bincount(mapping)[ir_idx].astype(float)
            ir_pos  = np.searchsorted(ir_idx, mapping)
            k_vecs = ir_grid[ir_idx] / np.array(nk_tuple, dtype=float) * 2*np.pi

            if not fine:
                self.k_vecs = k_vecs
                self.ir_idx = ir_idx
                self.ir_pos = ir_pos
                self.ibz_w_k = ibz_w_k
                self.full_k_vecs = full_k_vecs
            else:
                self.k_vecs_fine = k_vecs
                self.ir_idx_fine = ir_idx
                self.ir_pos_fine = ir_pos
                self.ibz_w_k_fine = ibz_w_k
                self.full_k_vecs_fine = full_k_vecs

    def get_e_k_Gf(self, nk):
        
        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)
        k_mesh = self.H_r.get_kmesh(nk_tuple)
        self.e_k_Gf = self.H_r.fourier(k_mesh)
    
    def get_e_k(self, fine=False):

        H_k = self.H_r.fourier(self.k_vecs)
        e_k = H_k[:,0,0].real

        if not fine:
            self.e_k = e_k
        else:
            self.e_k_fine = e_k

    def get_phase_k(self):
        
        self.phase_k = np.exp(1j * (self.R_vecs @ self.k_vecs.T))
        self.t_phase_k = self.t_vals[:, None] * self.phase_k

    def get_f_iwR(self, f_iwk, nk):
        if f_iwk.ndim == 1:
            f_iwk = f_iwk[None, :]

        dim = self.dim
        f_iwk_grid = f_iwk.reshape((f_iwk.shape[0],) + (nk,) * dim)
        f_iwR = np.fft.fftn(f_iwk_grid, axes=range(1, dim + 1)) / nk**dim

        return f_iwR

    def get_f_iwkq(self, f_iwk, q, nk, ibz=False, method='coarse', f_iwR=None):
        if f_iwk.ndim == 1:
            f_iwk = f_iwk[None, :]

        dim = self.dim
        q = np.array(q)

        if method == 'coarse':
            if ibz:
                f_iwk = self.unfold_f_k(f_iwk)

            f_iwk_grid = f_iwk.reshape((f_iwk.shape[0],) + (nk,) * dim)
            shifts = tuple(-round(qi * nk / 2) % nk for qi in q)
            f_iwkq_grid = np.roll(f_iwk_grid, shifts, axis=tuple(range(1, f_iwk_grid.ndim)))
            f_iwkq = f_iwkq_grid.reshape(f_iwk.shape)

            if ibz:
                f_iwkq = self.fold_f_k(f_iwkq)
        
            return f_iwkq

        elif method == 'fine':

            grid_1d = np.arange(nk)
            grids = np.meshgrid(*([grid_1d] * dim), indexing='ij')

            phase_q = np.ones((nk,) * dim, dtype=complex)
            for d in range(dim):
                phase_q *= np.exp(1j * np.pi * q[d] * grids[d])
            f_iwkq = np.fft.ifftn(
                f_iwR * phase_q[None, ...] * nk**dim,
                axes=range(1, dim + 1),
            )
            f_iwkq = f_iwkq.reshape(f_iwk.shape)

            return f_iwkq

    def get_e_kq(self, e_k, q, nk, ibz=False, method='coarse'):

        q = np.array(q)
        
        if method == 'fine':
            phase_q = np.exp(1j * np.pi * (self.R_vecs @ q[:self.dim]))
            
            e_kq = (phase_q @ self.t_phase_k).real
            de_kq_dq = np.array([
                (1j * np.pi * (self.R_vecs[:, alpha] * phase_q) @ self.t_phase_k).real
                for alpha in range(self.dim)
            ])
            
            return e_kq.real, de_kq_dq.real

        elif method == 'coarse':
            e_kq = self.get_f_iwkq(e_k, q, nk, ibz=ibz)
            return e_kq[0].real

    def unfold_e_k(self, e_k_ibz):
        return e_k_ibz[self.ir_pos]

    def fold_e_k(self, e_k_full):
        e_ibz = np.zeros(len(self.ir_idx))
        np.add.at(e_ibz, self.ir_pos, e_k_full)
        return e_ibz / self.ibz_w_k
    
    def unfold_f_k(self, f_iwk_ibz):
        return f_iwk_ibz[:, self.ir_pos]

    def fold_f_k(self, f_iwk_full):
        f_ibz = np.zeros((f_iwk_full.shape[0], len(self.ir_idx)))
        np.add.at(f_ibz, (slice(None), self.ir_pos), f_iwk_full)
        return f_ibz / self.ibz_w_k[None, :]