import numpy as np
from scripts.obs import sweep_rpa
from scripts.lattice import LATTICE, share_bz
from scripts.utils import merge_results
from h5 import HDFArchive
from mpi4py import MPI
import time, resource
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Gamma = 0.
tp = 0.
lat = LATTICE(tp=tp)
bz = share_bz(lat, nk=300, comm=comm)
bz_fine = share_bz(lat, nk=600, comm=comm)
#bz_fine = None

coarse = np.linspace(0.73, 1., 24)
fine = np.linspace(coarse[13], coarse[14], 10)
fine2 = np.linspace(coarse[1], coarse[4], 10)
#finer = np.linspace(coarse[10], coarse[12], 5)
#finer2 = np.linspace(coarse[12], coarse[13], 6)

coarse = coarse[((coarse < fine[0]) | (coarse > fine[-1])) & ((coarse < fine2[0]) | (coarse > fine2[-1]))]
#fine = fine[(fine < finer[0]) | (fine > finer[-1])]
#fine = fine[((fine < finer[0]) | (fine > finer[-1])) & ((fine < finer2[0]) | (fine > finer2[-1]))]

n_list = np.unique(np.concatenate([coarse, fine, fine2]))
T_list = np.linspace(0., 0.15, 50)
U = 3

file_name = f'G{Gamma:.5g}tp{tp:.5g}U{U:.5g}.h5'

if os.path.exists(f'data/diagram/{file_name}'):
    with HDFArchive(f'data/diagram/{file_name}', "r") as ar:
        if ar['fit'] is True:
            old_n = ar['n']

            Tc = - abs(ar['c']/ar['a'])**(1/ar['b']) * np.sign(ar['c']/ar['a'])

            # Fill NaNs by linear interpolation
            mask = np.isnan(Tc)
            Tc[mask] = np.interp(old_n[mask], old_n[~mask], Tc[~mask])

            old_Tc = np.interp(n_list, old_n, Tc)
        else:
            old_Tc = np.zeros(len(n_list))
        print(old_Tc)
else:
    old_Tc = np.zeros(len(n_list))

#new_T_list = T_list[:, np.newaxis] * (np.maximum(old_Tc[np.newaxis, :], 0.005)/0.005)**0.8 + np.maximum(old_Tc[np.newaxis, :], 0.005)
new_T_list = T_list[:, np.newaxis] + np.maximum(old_Tc[np.newaxis, :], 0.005)

par_list = [
    [{'U': U, 'n': n_list[i], 'T': new_T_list[j, i]}
     for j in range(len(T_list))]
    for i in range(len(n_list))
]

my_jobs = par_list[rank::size]
t0 = time.time()

print(f"rank {rank} got {len(my_jobs)} jobs")

results_list = []
for pars in my_jobs:
    results_list.append(sweep_rpa(pars, lat, bz, bz_fine, q_path=([1,1,0.5],[1,1,1]), method='local', fit_grid_pts=False, verbose=False, fit=True))
    #results_list.append(sweep_rpa(pars, lat, bz, bz_fine, niw=512, method='fft', S_list=-1j*0.025, verbose=False))

t1 = time.time()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"rank {rank} finished {len(my_jobs)} jobs in {t1 - t0:.2f} s | peak RAM = {peak:.1f} MB")

gathered = comm.gather(results_list, root=0)

if rank == 0:

    flattened = [r for sublist in gathered for r in sublist]

    flattened.sort(key=lambda d: d['n'])
    merged = merge_results(flattened)

    print(f"writing results to {file_name}")
    with HDFArchive(f'data/diagram/{file_name}', "w") as ar:
        for key, value in merged.items():
            ar[key] = value