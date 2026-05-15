import numpy as np
from triqs.gf import *
from triqs.plot.mpl_interface import *
from triqs_tprf.tight_binding import TBLattice
from triqs.lattice import BravaisLattice
from scripts.utils import *  

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
        #R_list = list(self.H_r.hoppings.keys())
        #R_list.sort()  # IMPORTANT
        #self.R_vecs = np.array([np.array(R) for R in R_list])
        #self.t_vals = np.array([self.H_r.hoppings[R][0][0] for R in R_list])
        
        self.phase_k = None
        self.e_k = None
        self.nk = None
        self.tetrahedra = None
    
    def get_k_mesh(self, nk, center=False):

        self.nk = nk

        nk_tuple = (nk,) * self.dim + (1,) * (3 - self.dim)

        k_mesh = self.H_r.get_kmesh(nk_tuple)

        k_arr = np.array([k.value for k in k_mesh])

        if center:
            k_arr = k_arr - np.pi

        return k_mesh, k_arr

    def get_e_k(self, nk, center=False):

        self.nk = nk
        
        k_mesh, _ = self.get_k_mesh(nk, center)

        self.e_k = self.H_r.fourier(k_mesh)

    def get_phase_k(self, nk):

        self.nk = nk

        _, k_arr = self.get_k_mesh(nk)
        k_arr = k_arr[:, :self.dim]
        
        self.phase_k = np.exp(1j * (self.R_vecs @ k_arr.T))

    def get_f_iwR(self, f_iwk):
        if f_iwk.ndim == 1:
            f_iwk = f_iwk[None, :]
        nk = self.nk
        dim = self.dim
        f_grid = f_iwk.reshape((f_iwk.shape[0],) + (nk,) * dim)
        return np.fft.fftn(f_grid, axes=range(1, dim + 1)) / nk**dim
        # returns shape (niw, nk, nk[, nk])

    def get_f_iwkq(self, f_iwk, q, method='coarse', f_iwR=None):
        if f_iwk.ndim == 1:
            f_iwk = f_iwk[None, :]
        dim = self.dim
        nk = self.nk
        q = np.array(q)

        if method == 'coarse':
            f_iwk_grid = f_iwk.reshape((f_iwk.shape[0],) + (nk,) * dim)
            shifts = tuple(-round(qi * nk / 2) % nk for qi in q)
            f_iwkq = np.roll(f_iwk_grid, shifts, axis=tuple(range(1, f_iwk_grid.ndim)))
            return f_iwkq.reshape(f_iwk.shape)

        elif method == 'fine':
            grid_1d = np.arange(nk)
            grids = np.meshgrid(*([grid_1d] * dim), indexing='ij')
            phase_q = np.ones((nk,) * dim, dtype=complex)
            for d in range(dim):
                phase_q *= np.exp(1j * np.pi * q[d] * grids[d])
            f_kq = np.fft.ifftn(f_iwR * phase_q[None, ...] * nk**dim,
                                axes=range(1, dim + 1))
            return f_kq.reshape(f_iwk.shape)
    
    def get_e_kq(self, e_k, q, method='coarse'):

        if method == 'fine':
            phase_q = np.exp(1j * np.pi * (self.R_vecs @ np.array(q)[:self.dim]))
            e_kq = (self.t_vals * phase_q) @ self.phase_k
            return e_kq.real
        
        elif method == 'coarse':
            e_kq = self.get_f_iwkq(e_k, q, method)
            return e_kq[0].real
        
    def get_tetrahedra(self, nk):
        self.nk = nk
        assert self.dim == 3

        def idx(i, j, k):
            return (i % nk)*nk*nk + (j % nk)*nk + (k % nk)

        tets = []
        for i in range(nk):          # full range
            for j in range(nk):
                for k in range(nk):
                    v000 = idx(i,   j,   k  )
                    v100 = idx(i+1, j,   k  )
                    v010 = idx(i,   j+1, k  )
                    v110 = idx(i+1, j+1, k  )
                    v001 = idx(i,   j,   k+1)
                    v101 = idx(i+1, j,   k+1)
                    v011 = idx(i,   j+1, k+1)
                    v111 = idx(i+1, j+1, k+1)
                    tets += [
                        [v000, v100, v010, v001],
                        [v100, v110, v010, v111],
                        [v100, v010, v001, v111],
                        [v010, v001, v011, v111],
                        [v100, v001, v101, v111],
                        [v001, v011, v101, v111],
                    ]

        self.tetrahedra = np.array(tets)