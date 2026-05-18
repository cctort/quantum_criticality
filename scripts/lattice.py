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
        
        self.phase_k = None
        self.e_k_Gf = None
        self.e_k = None
        self.e_k_fine = None

        self.ir_idx = None
        self.ir_pos = None
        self.weights = None
        self.ir_idx_fine = None
        self.ir_pos_fine = None
        self.weights_fine = None

    def build_ibz_maps(self, nk, fine=False):

        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)
        mapping, _ = spglib.get_ir_reciprocal_mesh(nk_tuple, self._spg_cell, is_shift=[0,0,0])

        if not fine:
            self.ir_idx  = np.unique(mapping)
            self.ir_pos  = np.searchsorted(self.ir_idx, mapping)
            self.weights = np.bincount(mapping)[self.ir_idx]
        else:
            self.ir_idx_fine  = np.unique(mapping)
            self.ir_pos_fine  = np.searchsorted(self.ir_idx_fine, mapping)
            self.weights_fine = np.bincount(mapping)[self.ir_idx_fine]
    
    def get_k_mesh(self, nk):

        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)
        k_mesh = self.H_r.get_kmesh(nk_tuple)
        k_arr = np.array([k.value for k in k_mesh]) - np.pi

        return k_arr
    
    def get_k_mesh_reduced(self, nk):
        
        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)
        kvecs = self.ir_grid / np.array(nk_tuple, dtype=float)

    def get_e_k_old(self, nk, center=False, fine=False, keep_Gf=False):
        
        k_mesh, _ = self.get_k_mesh(nk, center)
        self.e_k_Gf = self.H_r.fourier(k_mesh)
        if not fine:
            self.e_k = self.e_k_Gf.data[:,0,0].real.copy()
        else:
            self.e_k_fine = self.e_k_Gf.data[:,0,0].real.copy()

        if not keep_Gf:
            del self.e_k_Gf
    
    def get_e_k(self, nk, fine=False):

        H_k = self.H_r.fourier(kvecs)
        e_ibz = H_k[:,0,0].real

        if not fine:
            self.e_k = e_ibz
        else:
            self.e_k_fine = e_ibz

    def get_phase_k(self, nk):
        
        _, k_arr = self.get_k_mesh(nk)
        k_arr = k_arr[:, :self.dim]
        
        self.phase_k = np.exp(1j * (self.R_vecs @ k_arr.T))
        self.t_phase_k = self.t_vals[:, None] * self.phase_k

    def get_f_iwR(self, f_iwk, nk):
        if f_iwk.ndim == 1:
            f_iwk = f_iwk[None, :]
        dim = self.dim
        f_grid = f_iwk.reshape((f_iwk.shape[0],) + (nk,) * dim)
        return np.fft.fftn(f_grid, axes=range(1, dim + 1)) / nk**dim

    def get_f_iwkq(self, f_iwk, q, nk, method='coarse', f_iwR=None):
        if f_iwk.ndim == 1:
            f_iwk = f_iwk[None, :]
        dim = self.dim
        q = np.array(q)

        if method == 'coarse':
            f_iwk_grid = f_iwk.reshape((f_iwk.shape[0],) + (nk,) * dim)
            shifts = tuple(-round(qi * nk / 2) % nk for qi in q)
            f_iwkq_grid = np.roll(f_iwk_grid, shifts, axis=tuple(range(1, f_iwk_grid.ndim)))
        
            return f_iwkq_grid.reshape(f_iwk.shape)

        elif method == 'fine':
            grid_1d = np.arange(nk)
            grids = np.meshgrid(*([grid_1d] * dim), indexing='ij')
            phase_q = np.ones((nk,) * dim, dtype=complex)
            for d in range(dim):
                phase_q *= np.exp(1j * np.pi * q[d] * grids[d])
            f_kq = np.fft.ifftn(
                f_iwR * phase_q[None, ...] * nk**dim,
                axes=range(1, dim + 1),
            )
            f_kq = f_kq.reshape(f_iwk.shape)
            return f_kq

    def get_e_kq(self, e_k, q, nk, method='coarse'):
        
        if method == 'fine':
            phase_q = np.exp(1j * np.pi * (self.R_vecs @ np.array(q)[:self.dim]))
            
            e_kq = (phase_q @ self.t_phase_k).real
            de_kq_dq = np.array([
                (1j * np.pi * (self.R_vecs[:, alpha] * phase_q) @ self.t_phase_k).real
                for alpha in range(self.dim)
            ])
            
            return e_kq.real, de_kq_dq.real

        elif method == 'coarse':
            e_k_mesh = e_k[np.newaxis, :] if e_k.ndim == 1 else e_k
            e_kq = self.get_f_iwkq(e_k_mesh, q, nk, method)
            return e_kq[0].real
    
    def unfold_e_k(self, e_k_ibz):
        return e_k_ibz[self.ir_pos]

    def fold_e_k(self, e_k_full):
        return e_k_full[self.ir_idx]