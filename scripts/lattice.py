import numpy as np
from triqs.gf import *
from triqs.plot.mpl_interface import *
from triqs_tprf.tight_binding import TBLattice
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
        
        self.phase_k = None
        self.e_k = None
        self.nk = None

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
        
        self.phase_k = np.exp(1j * (k_arr @ self.R_vecs.T))

    def get_e_kq(self, q, method):
    
        dim = self.dim
        nk = self.nk

        q = np.array(q)
        phase_q = np.exp(1j * np.pi * (self.R_vecs @ q))
        
        e_k = self.e_k.data[:,0,0].real

        if method == 'coarse':
            e_k_grid = e_k.reshape((nk,) * dim)
            shifts  = tuple(-round(qi * nk / 2) % nk for qi in q)
            e_kq = np.roll(e_k_grid, shifts, axis=tuple(range(dim))).ravel()

        elif method == 'fine':
            e_kq = (self.phase_k @ (self.t_vals * phase_q)).real
        
        return e_k, e_kq
  