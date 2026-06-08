import numpy as np
from triqs.gf import *
from triqs.plot.mpl_interface import *
from triqs_tprf.tight_binding import TBLattice
from scripts.utils import *  
import spglib

class LATTICE:
    """
    Class containing lattice informations
    """
    def __init__(self, t=1., tp=0., dim=3):

        self.t   = t
        self.tp  = tp
        self.dim = dim

        self.units    = []
        self.hoppings = {}

        nn_vecs = []
        for i in range(dim):
            ax = tuple(int(j==i) for j in range(dim))
            nax = tuple(-x for x in ax)
            self.units += [ax]
            nn_vecs += [ax, nax]

        if t != 0:
            for vec in nn_vecs:
                self.hoppings[vec] = [[-t]]

        if dim == 2:
            nnn_vecs = [(1,1),(1,-1),(-1,1),(-1,-1)]
        elif dim == 3:
            nnn_vecs = [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                        (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                        (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]

        if tp != 0:
            for vec in nnn_vecs:
                self.hoppings[vec] = [[-tp]]

        self.a_vecs = np.array(self.units).T
        self.b_vecs = 2*np.pi * np.linalg.inv(self.a_vecs).T

        H_r = TBLattice(units=self.units, hoppings=self.hoppings)

        self.t_vals = np.array([t[0,0] for t in H_r.hoppings.values()])
        self.R_vecs = np.array([np.array(R[:dim]) for R in H_r.hoppings])

        self.R_vecs_NN  = np.array(nn_vecs)
        self.R_vecs_NNN = np.array(nnn_vecs)
        
        unit_mat = np.eye(3)
        unit_mat[:dim, :dim] = np.array(self.units, dtype=float)
        self.spg_cell = (unit_mat, [[0., 0., 0.]], [1])

        self.nk, self.nk_fine = None, None
        self.k_vecs, self.k_vecs_fine = None, None
        self.e_k, self.e_k_fine = None, None

        self.ibz_w_k, self.ibz_w_k_fine = None, None
        self.ibz_idx, self.ibz_idx_fine = None, None
        self.ibz_pos, self.ibz_pos_fine = None, None

    def get_bz(self, nk, ibz=False, fine=False):

        if not fine:
            self.nk = nk
        else:
            self.nk_fine = nk

        self.ibz = ibz

        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)
        H_r = TBLattice(units=self.units, hoppings=self.hoppings)

        if not ibz:
            k_mesh = H_r.get_kmesh(nk_tuple)
            k_vecs = np.array([k.value for k in k_mesh])
            ibz_w_k = np.ones(nk**self.dim)

            if not fine:
                self.k_vecs, self.ibz_w_k = np.ascontiguousarray(k_vecs), ibz_w_k
            else:
                self.k_vecs_fine, self.ibz_w_k_fine = np.ascontiguousarray(k_vecs), ibz_w_k

        else:
            mapping, ir_grid = spglib.get_ir_reciprocal_mesh(nk_tuple, self.spg_cell)

            ibz_idx = np.unique(mapping)
            ibz_w_k = np.bincount(mapping)[ibz_idx].astype(float)
            ibz_pos  = np.searchsorted(ibz_idx, mapping)
            k_vecs = (ir_grid[ibz_idx] / np.array(nk_tuple, dtype=float) * 2*np.pi)[:,:self.dim]

            if not fine:
                self.k_vecs, self.ibz_w_k = np.ascontiguousarray(k_vecs), ibz_w_k
                self.ibz_idx, self.ibz_pos = ibz_idx, ibz_pos
            else:
                self.k_vecs_fine, self.ibz_w_k_fine = np.ascontiguousarray(k_vecs), ibz_w_k
                self.ibz_idx_fine, self.ibz_pos_fine = ibz_idx, ibz_pos
    
    def get_e_k(self, fine=False):

        H_r = TBLattice(units=self.units, hoppings=self.hoppings)

        if not fine:
            H_k = H_r.fourier(self.k_vecs/(2*np.pi))
            self.e_k = H_k[:,0,0].real
        else:
            H_k = H_r.fourier(self.k_vecs_fine/(2*np.pi))
            self.e_k_fine = H_k[:,0,0].real

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
    
def get_f_R(f_k, nk, dim):

    f_k_grid = f_k.reshape((nk,) * dim)
    f_R = np.fft.fftn(f_k_grid) / nk**dim

    return f_R

def get_f_iwR(f_iwk, nk, dim):

    niw = f_iwk.shape[0]
    f_iwk_grid = f_iwk.reshape((niw,) + (nk,) * dim)
    f_iwR = np.fft.fftn(f_iwk_grid, axes=range(1, dim + 1)) / nk**dim

    return f_iwR

def get_f_iwkq(f_iwk, q, nk, method='roll', f_iwR=None):
    
    q = np.array(q)
    dim = len(q)
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

def get_e_kq(e_k, q, nk, method='roll', R_vecs=None, t_vals=None):
    dim = len(q)
    q = np.array(q)
    Nk = len(e_k)

    if method == 'roll':
        e_k_grid = e_k.reshape((nk,) * dim)
        shifts = tuple(-round(qi * nk / 2) % nk for qi in q)
        e_kq = np.roll(e_k_grid, shifts, axis=tuple(range(dim))).reshape(-1)
        return e_kq

    elif method == 'exact':
        phase_q = np.exp(1j * np.pi * (R_vecs @ q))
        t_phase_q = t_vals * phase_q

        R_grid = np.zeros((nk,) * dim, dtype=complex)
        for tq, R in zip(t_phase_q, R_vecs):
            R_grid[tuple(int(r) % nk for r in R)] += tq
        e_kq = np.fft.ifftn(R_grid).real.reshape(-1) * Nk

        de_kq_dq = np.zeros((dim, Nk))
        for alpha in range(dim):
            dR_grid = np.zeros((nk,) * dim, dtype=complex)
            for tq, R in zip(1j * np.pi * R_vecs[:, alpha] * t_phase_q, R_vecs):
                dR_grid[tuple(int(r) % nk for r in R)] += tq
            de_kq_dq[alpha] = np.fft.ifftn(dR_grid).real.reshape(-1) * Nk

        return e_kq, de_kq_dq