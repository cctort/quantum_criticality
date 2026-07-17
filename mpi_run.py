import numpy as np
from mpi4py import MPI
from scripts.obs import run_rpa
from scripts.lattice import LATTICE, share_bz
from scripts.utils import merge_results
from h5 import HDFArchive
import time, resource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tp = 0.
lat = LATTICE(tp=tp)
bz = share_bz(lat, nk=400, comm=comm)
bz_fine = bz

n = 0.745
T_list = np.linspace(0.005, 0.2, 21)
U = 3

file_name = f'tp{tp:.5g}n{n:.5g}U{U:.5g}.h5'
print(f'will write to {file_name}')

par_list = [{'U': U, 'n': n, 'T': T} for T in T_list]
my_jobs  = par_list[rank::size]
t0 = time.time()

results_list = []
for par in my_jobs:
    results_list.append(run_rpa(par, lat, bz, bz_fine, q_path=([1,1,0.5],[1,1,1.]),
                                method='local', always_fit_qmin=True,
                                fit_grid_pts=False, verbose=False))
    
    
t1 = time.time()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"rank {rank} finished {len(my_jobs)} jobs in {t1 - t0:.2f} s | peak RAM = {peak:.1f} MB")

gathered = comm.gather(results_list, root=0)

if rank == 0:
    flattened = [r for sublist in gathered for r in sublist]
    flattened.sort(key=lambda d: d['T'])
    merged = merge_results(flattened, ['invchi', 'invchi_min', 'invxi_min',
                           'Q', 'OZ_fit', 'OZ_weight', 'Q_fitted', 'mu'])

    print(f"writing results to {file_name}")
    with HDFArchive(f'data/chimin/{file_name}', "w") as ar:
        for key, value in merged.items():
            ar[key] = value