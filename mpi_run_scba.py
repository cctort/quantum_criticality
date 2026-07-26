import numpy as np
from mpi4py import MPI
from scripts.scba import SCBA
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
bz = share_bz(lat, nk=500, comm=comm)
bz_fine = bz

n = 0.8374
T_list = np.linspace(0.005, 0.2, 21)
U = 3
v = 0.3
niw = 2048

file_name = f'v{v:.5g}tp{tp:.5g}n{n:.5g}U{U:.5g}.h5'

par_list = [{'U': U, 'n': n, 'T': T} for T in T_list]
my_jobs  = par_list[rank::size]

scba = SCBA(lat, bz, niw)

t0 = time.time()

results_list = []
S_prev = None
for par in my_jobs:

    scba_result = scba.run(par, v=v, init_S=S_prev, max_iter=500, tol=1e-10, mix=0., verbose=False)
    S_prev = scba_result['S_iwk']

    S_val = scba_result['S_iwk'].ravel()
    results = run_rpa(par, lat, bz, None, q_path=([1,1,0.5],[1,1,1.]), xi_range=[0,0,1e-2], niw=niw, S_val=S_val, verbose=False)

    results['diff'] = scba_result['diff']
    results['converged'] = scba_result['converged']

    results_list.append(results)

t1 = time.time()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"rank {rank} finished {len(my_jobs)} jobs in {t1 - t0:.2f} s | peak RAM = {peak:.1f} MB")

gathered = comm.gather(results_list, root=0)

if rank == 0:
    flattened = [r for sublist in gathered for r in sublist]
    flattened.sort(key=lambda d: d['T'])
    merged = merge_results(flattened, ['invchi', 'invchi_min', 'invxi_min', 'Q', 'OZ_fit',
                                        'OZ_weight', 'Q_fitted', 'mu',
                                        'scba_diff', 'scba_converged'])

    print(f"writing results to {file_name}")
    with HDFArchive(f'data/chimin/{file_name}', "w") as ar:
        for key, value in merged.items():
            ar[key] = value