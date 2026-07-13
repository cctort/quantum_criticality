#!/bin/bash
#SBATCH --job-name=rpa
#SBATCH --partition=bilbao
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=12
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=log/rpa_%j.out

eval "$(/home/carlo/tools/bin/micromamba shell hook --shell bash)"
micromamba activate triqs_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_THREADING_LAYER=omp

export PMIX_MCA_psec=native
mkdir -p memray_out
srun --mpi=pmix python mpi_scaling.py
#py-spy record --native -o profile_native.svg --rate 20 --subprocesses -- python mpi_scaling.py
#srun --mpi=pmix bash -c '
#  memray run --aggregate -f -o memray_out/profile_rank${SLURM_PROCID}.bin -- mpi_scaling.py
#'
#mpirun -np 5 python mpi_scaling.py