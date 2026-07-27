import numpy as np
from mpi4py import MPI
from scripts.scba import SCBA
from scripts.obs import sweep_rpa
from scripts.lattice import LATTICE, share_bz
from scripts.utils import merge_results
from h5 import HDFArchive
import time, resource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tp = 0.
lat = LATTICE(tp=tp)
bz = share_bz(lat, nk=500, comm=comm)
bz_fine = None

#n_list = np.linspace(0.785, 0.789, 5) # v = 0.1
n_list = np.linspace(0.8368, 0.8378, 5) # v = 0.3
T_list = np.linspace(0.005, 0.015, 15)
U = 3
v = 0.3
niw = 2048

file_name = f'v{v:.5g}tp{tp:.5g}U{U:.5g}.h5'

par_list = [[{'U': U, 'n': n, 'T': T} for T in T_list] for n in n_list]

my_jobs = par_list[rank::size]
t0 = time.time()

print(f"rank {rank} got {len(my_jobs)} jobs")

scba = SCBA(lat, bz, niw)

results_list = []
for pars in my_jobs:
    
    S_list = []
    scba_diff = []
    scba_converged = []
    S_prev = None
    for par in pars:
        scba_result = scba.run(par, v=v, init_S=S_prev, max_iter=500, tol=1e-10, mix=0., verbose=False)
        S_prev = scba_result['S_iwk']
        S_list.append(S_prev.ravel())
        scba_diff.append(scba_result['diff'])
        scba_converged.append(scba_result['converged'])

    results = sweep_rpa(pars, lat, bz, bz_fine, niw=niw, method='fft',
                         S_list=S_list, verbose=False, xi_range=[0, 0, 1e-2])

    results['scba_diff'] = np.array(scba_diff)
    results['scba_converged'] = np.array(scba_converged)

    results_list.append(results)

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