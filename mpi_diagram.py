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
bz = share_bz(lat, nk=400, comm=comm)
bz_fine = share_bz(lat, nk=800, comm=comm)
#bz_fine = None

coarse = np.linspace(0.73, 1., 24)
fine = np.linspace(0.884, 0.89, 8)
finer = np.linspace(0.887, 0.889, 6)

coarse = coarse[(coarse < fine[0]) | (coarse > fine[-1])]
fine = fine[(fine < finer[0]) | (fine > finer[-1])]

n_list = np.sort(np.concatenate([coarse, fine, finer]))
T_list = np.linspace(0., 0.2, 100)
U = 3

file_name = f'G{Gamma:.5g}tp{tp:.5g}U{U:.5g}.h5'

if os.path.exists(f'data/diagram/{file_name}'):
    with HDFArchive(f'data/diagram/{file_name}', "r") as ar:
        if ar['fit'] is True:
            old_n = ar['n']

            Tc = - abs(ar['c']/ar['a'])**(1/ar['b']) * np.sign(ar['c']/ar['a'])

            # Fill NaNs by linear interpolation
            mask = np.isnan(Tc)
            Tc[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                Tc[~mask]
            )

            old_Tc = np.interp(n_list, ar['n'], Tc)
        else:
            old_Tc = np.zeros(len(n_list))
else:
    old_Tc = np.zeros(len(n_list))

par_list = [[{'U': U, 'n': n_list[i], 'T': T + max(old_Tc[i], 0.005)} for T in T_list] for i in range(len(n_list))]

my_jobs = par_list[rank::size]
t0 = time.time()

print(f"rank {rank} got {len(my_jobs)} jobs")

results_list = []
for pars in my_jobs:
    results_list.append(sweep_rpa(pars, lat, bz, bz_fine, q_path=([1,1,0.5],[1,1,1]), method='local', fit_grid_pts=False, verbose=False, xi_range=[0,0,1e-2], fit=True))
    #results_list.append(sweep_rpa(pars, lat, bz, bz_fine, niw=512, method='fft', S_list=-1j*0.025, verbose=False, xi_range=[0,0,2e-2]))

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