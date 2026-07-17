import numpy as np
import scipy.fft as spfft
from triqs.gf import *
from triqs.plot.mpl_interface import *
from triqs_tprf.tight_binding import TBLattice
from scripts.utils import *  
import spglib
import numpy as np
from mpi4py import MPI
import gc
import os
cpw = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

class LATTICE:

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

    def to_dict(self):
        return {'t': self.t, 'tp': self.tp, 'dim': self.dim}

class BZ:
    
    def __init__(self, lat, nk, ibz=True, store_full_bz=False):
        
        self.nk = nk
        self.dim = lat.dim
        self.ibz = ibz
        Nk = nk**self.dim

        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)
        H_r = TBLattice(units=lat.units, hoppings=lat.hoppings)

        k_mesh = H_r.get_kmesh(nk_tuple)
        self.k_full = np.asarray([k.value for k in k_mesh]) if store_full_bz else None

        if ibz:
            mapping, ir_grid = spglib.get_ir_reciprocal_mesh(nk_tuple, lat.spg_cell)

            ibz_idx = np.unique(mapping)
            w_k = np.bincount(mapping)[ibz_idx].astype(float)/Nk
            ibz_pos  = np.searchsorted(ibz_idx, mapping)
            k_vecs = (ir_grid[ibz_idx] / np.array(nk_tuple, dtype=float) * 2*np.pi)[:,:self.dim]
            
            self.k_vecs = k_vecs
            self.w_k, self.ibz_idx, self.ibz_pos = w_k, ibz_idx, ibz_pos
        
        else:
            k_mesh = H_r.get_kmesh(nk_tuple)
            k_vecs = np.asarray([k.value for k in k_mesh])

            self.w_k = np.ones(Nk)/Nk
            self.k_vecs = k_vecs

        H_k = H_r.fourier(k_vecs/(2*np.pi))
        self.e_k = H_k[:,0,0].real
    
    def to_dict(self):
        return {'nk': self.nk, 'ibz': self.ibz}

    def unfold_f_iwk(self, f_iwk_ibz):
        return f_iwk_ibz[:, self.ibz_pos]

    def fold_f_iwk(self, f_iwk_full):
        return f_iwk_full[:, self.ibz_idx]
    
    def unfold_f_k(self, f_k_ibz):
        return f_k_ibz[self.ibz_pos]

    def fold_f_k(self, f_k_full):
        return f_k_full[self.ibz_idx]

def get_f_R(f_k, nk, dim):

    f_k_grid = f_k.reshape((nk,) * dim)
    f_R = spfft.fftn(f_k_grid, workers=cpw, overwrite_x=True) / nk**dim

    return f_R

def get_f_iwR(f_iwk, nk, dim):

    niw = f_iwk.shape[0]
    f_iwk_grid = f_iwk.reshape((niw,) + (nk,) * dim)
    f_iwR = spfft.fftn(f_iwk_grid, axes=range(1, dim + 1), workers=cpw, overwrite_x=True) / nk**dim

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

    elif method == 'fft':
        grid_1d = np.arange(nk)
        phase_grid = np.ones((nk,) * dim, dtype=complex)
        for alpha in range(dim):
            shape = [1] * dim
            shape[alpha] = nk
            phase_grid *= np.exp(1j * np.pi * q[alpha] * grid_1d.reshape(shape))

        f_iwkq = spfft.ifftn(f_iwR * phase_grid[None, ...], axes=range(1, dim + 1), workers=cpw, overwrite_x=True).reshape(f_iwk.shape) * Nk

        df_iwkq_dq = np.zeros((dim,) + f_iwk.shape)   # (dim, niw, nk^dim)
        for alpha in range(dim):
            shape = [1] * dim
            shape[alpha] = nk
            d_phase_grid = (1j * np.pi * grid_1d.reshape(shape)) * phase_grid
            df_iwkq_dq[alpha] = spfft.ifftn(f_iwR * d_phase_grid[None, ...], axes=range(1, dim + 1), workers=cpw, overwrite_x=True).reshape(f_iwk.shape) * Nk

        return f_iwkq, df_iwkq_dq

def _get_scatter_idx_rfft(R_vecs, t_vals, nk, dim):
    nk_r = nk // 2 + 1
    R_mod = R_vecs.astype(np.int64) % nk
    keep = R_mod[:, -1] <= nk // 2          # one of each ±R pair
    flat_idx = np.ravel_multi_index(
        tuple(R_mod[keep].T[:-1]) + (R_mod[keep, -1],),
        dims=(nk,) * (dim - 1) + (nk_r,)
    )
    return flat_idx, t_vals[keep], R_vecs[keep]

def get_e_kq(e_k, q, nk, method='roll', R_vecs=None, t_vals=None, k_full=None):
    dim = len(q)
    q = np.array(q)
    Nk = len(e_k)

    if method == 'roll':
        e_k_grid = e_k.reshape((nk,) * dim)
        shifts = tuple(-round(qi * nk / 2) % nk for qi in q)
        e_kq = np.roll(e_k_grid, shifts, axis=tuple(range(dim))).reshape(-1)

    elif method == 'exact':
        kq = k_full + q
        phase_kq = np.exp(1j * np.pi * (kq @ R_vecs.T))
        e_kq = np.sum(t_vals * phase_kq, axis=1)

    elif method == 'fft':
        flat_idx, t_vals_r, R_vecs_r = _get_scatter_idx_rfft(R_vecs, t_vals, nk, dim)

        phase_q = np.exp(1j * np.pi * (R_vecs_r @ q))
        t_phase_q = t_vals_r * phase_q

        R_grid_flat = np.zeros((nk,) * (dim - 1) + (nk // 2 + 1,), dtype=np.complex128)
        e_kq = np.empty(Nk, dtype=np.float64)

        np.add.at(R_grid_flat.reshape(-1), flat_idx, t_phase_q)

        res = spfft.irfftn(R_grid_flat, s=(nk,) * dim, workers=cpw)
        np.multiply(res.reshape(-1), Nk, out=e_kq)

    return e_kq


def get_de_kq_dq(e_k, q, nk, R_vecs, t_vals, method='fft', k_full=None):
    dim = len(q)
    q = np.array(q)
    Nk = len(e_k)

    de_kq_dq = np.zeros((dim, Nk), dtype=np.float64)

    if method == 'fft':
        flat_idx, t_vals_r, R_vecs_r = _get_scatter_idx_rfft(R_vecs, t_vals, nk, dim)

        phase_q = np.exp(1j * np.pi * (R_vecs_r @ q))
        t_phase_q = t_vals_r * phase_q

        grid_1d = np.empty((nk,) * (dim - 1) + (nk // 2 + 1,), dtype=np.complex128)

        for d in range(dim):
            grid_1d.fill(0)
            weighted = (1j * np.pi * R_vecs_r[:, d]) * t_phase_q
            np.add.at(grid_1d.reshape(-1), flat_idx, weighted)

            res = spfft.irfftn(grid_1d, s=(nk,) * dim, workers=cpw)
            np.multiply(res.reshape(-1), Nk, out=de_kq_dq[d])

    else:
        kq = k_full + q
        phase_kq = np.exp(1j * np.pi * (kq @ R_vecs.T))
        t_phase_kq = t_vals * phase_kq

        for d in range(dim):
            de_kq_dq[d] = np.sum(
                1j * np.pi * R_vecs[:, d] * t_phase_kq,
                axis=1
            ).real

    return de_kq_dq

def share_bz(lat, nk, comm, ibz=True, store_full_bz=False):
    comm_node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = comm_node.Get_rank()

    if node_rank == 0:
        bz0 = BZ(lat, nk=nk, ibz=ibz, store_full_bz=store_full_bz)
        N_ibz = len(bz0.e_k)
        N_full = len(bz0.ibz_pos) if ibz else 0
        meta = (N_ibz, N_full, bz0.nk, bz0.dim, ibz, store_full_bz)
    else:
        bz0 = meta = None

    N_ibz, N_full, nk, dim, ibz, store_full_bz = comm_node.bcast(meta, root=0)

    n_kvecs = N_ibz * dim
    n_kfull = N_full * dim if store_full_bz else 0
    n_wk    = N_ibz if ibz else 0
    n_f64 = N_ibz + n_kvecs + n_kfull + n_wk
    n_i64 = (N_ibz + N_full) if ibz else 0

    win_f = MPI.Win.Allocate_shared(n_f64 * 8 if node_rank == 0 else 0, 8, comm=comm_node)
    win_i = MPI.Win.Allocate_shared(n_i64 * 8 if node_rank == 0 else 0, 8, comm=comm_node)

    flat_f = np.ndarray(n_f64, dtype=np.float64, buffer=win_f.Shared_query(0)[0])
    flat_i = np.ndarray(n_i64, dtype=np.int64, buffer=win_i.Shared_query(0)[0]) if n_i64 else None

    s_e_k    = slice(0, N_ibz)
    s_kvecs  = slice(N_ibz, N_ibz + n_kvecs)
    s_kfull  = slice(N_ibz + n_kvecs, N_ibz + n_kvecs + n_kfull)
    s_wk     = slice(N_ibz + n_kvecs + n_kfull, N_ibz + n_kvecs + n_kfull + n_wk)
    s_ibzidx = slice(0, N_ibz)
    s_ibzpos = slice(N_ibz, N_ibz + N_full)

    if node_rank == 0:
        flat_f[s_e_k] = bz0.e_k
        flat_f[s_kvecs] = bz0.k_vecs.ravel()
        if store_full_bz:
            flat_f[s_kfull] = bz0.k_full.ravel()
        if ibz:
            flat_f[s_wk] = bz0.w_k
            flat_i[s_ibzidx] = bz0.ibz_idx
            flat_i[s_ibzpos] = bz0.ibz_pos
        del bz0
        gc.collect()

    comm_node.Barrier()

    bz = BZ.__new__(BZ)
    bz.nk  = nk
    bz.dim = dim
    bz.ibz = ibz
    bz.e_k     = flat_f[s_e_k]
    bz.k_vecs  = flat_f[s_kvecs].reshape(N_ibz, dim)
    bz.k_full  = flat_f[s_kfull].reshape(N_full, dim) if store_full_bz else None
    bz.w_k     = flat_f[s_wk] if ibz else None
    bz.ibz_idx = flat_i[s_ibzidx] if ibz else None
    bz.ibz_pos = flat_i[s_ibzpos] if ibz else None
    bz._wins = (win_f, win_i, comm_node)

    return bz