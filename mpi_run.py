import os, numpy as np
from scripts.obs import run_rpa
from scripts.lattice import LATTICE, BZ
from scripts.utils import merge_results
from h5 import HDFArchive
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 0.848
T_list = np.linspace(0.005, 0.2, 21)
U = 3

lat = LATTICE()
bz = BZ(lat, nk=50)
bz_fine = BZ(lat, nk=100)

par_list = [{'U': U, 'n': n, 'T': T} for T in T_list]

my_jobs = par_list[rank::size]

print(f"rank {rank} got {len(my_jobs)} jobs")

results_list = []
for par in my_jobs:
    results_list.append(run_rpa(par, lat, bz, bz_fine, q_path=([1,1,0],[1,1,1]), method='local', always_fit_qmin=True, verbose=False))

gathered = comm.gather(results_list, root=0)

if rank == 0:

    flattened = [r for sublist in gathered for r in sublist]
    merged = merge_results(flattened)

    with HDFArchive("results.h5", "w") as ar:
        for key, value in merged.items():
            ar[key] = value