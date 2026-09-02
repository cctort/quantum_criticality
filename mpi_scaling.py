import numpy as np
from scripts.obs import sweep_rpa
from scripts.lattice import LATTICE, share_bz
from scripts.utils import merge_results
from h5 import HDFArchive
from mpi4py import MPI
import time, resource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Gamma = 0.005
tp = 0.
lat = LATTICE(tp=tp)
bz = share_bz(lat, nk=500, comm=comm)
#bz_fine = share_bz(lat, nk=1000, comm=comm)
#bz_fine = None
bz_fine = bz

#n_list = np.linspace(0.735, 0.755, 5)
#n_list = np.linspace(0.846, 0.85, 5) # tp=0.1
#n_list = np.linspace(0.884905, 0.884965, 5) # tp=0.2
n_list = np.linspace(0.775, 0.785, 5) # Gamma=0.005
#n_list = np.linspace(0.795, 0.798, 5) # Gamma=0.01
#n_list = np.linspace(0.805, 0.81, 5) # Gamma=0.015
#n_list = np.linspace(0.814, 0.818, 5) # Gamma=0.02 - done, go up
#n_list = np.linspace(0.835, 0.837, 5) # Gamma=0.035
#n_list = np.linspace(0.8503, 0.8513, 5) # Gamma=0.05
#n_list = np.linspace(0.8846, 0.8851, 5) # Gamma=0.1
#n_list = np.linspace(0.8598, 0.8608, 5) # Gamma=0.01, tp=0.1
#n_list = np.linspace(0.8857, 0.8859, 5) # Gamma=0.05, tp=0.1
#n_list = np.linspace(0.9055, 0.9056, 5) # Gamma=0.1, tp=0.1
#n_list = np.linspace(0.8874, 0.88744, 5) # Gamma=0.01, tp=0.2
#n_list = np.linspace(0.8975, 0.89754, 5) # Gamma=0.05, tp=0.2
#T_list = np.linspace(0.005, 0.02, 20)
T_list = np.arange(0.005, 0.02, 0.00075)
U = 3

file_name = f'G{Gamma:.5g}tp{tp:.5g}U{U:.5g}.h5'

par_list = [[{'U': U, 'n': n, 'T': T} for T in T_list] for n in n_list]

my_jobs = par_list[rank::size]
t0 = time.time()

print(f"rank {rank} got {len(my_jobs)} jobs")

results_list = []
for pars in my_jobs:
    #results_list.append(sweep_rpa(pars, lat, bz, bz_fine, q_path=([1,1,0.5],[1,1,1]), method='local', fit_grid_pts=False, fit=True, verbose=False))
    results_list.append(sweep_rpa(pars, lat, bz, bz_fine, niw=2048, method='fft', S_list=-1j*Gamma, fit_grid_pts=False, fit=True, verbose=False))

t1 = time.time()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"rank {rank} finished {len(my_jobs)} jobs in {t1 - t0:.2f} s | peak RAM = {peak:.1f} MB")

gathered = comm.gather(results_list, root=0)

if rank == 0:

    flattened = [r for sublist in gathered for r in sublist]

    flattened.sort(key=lambda d: d['n'])
    merged = merge_results(flattened)

    print(f"writing results to {file_name}")
    with HDFArchive(f'data/scaling/{file_name}', "w") as ar:
        for key, value in merged.items():
            ar[key] = value