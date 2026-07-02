#!/bin/bash
#SBATCH --job-name=rpa
#SBATCH --partition=bilbao
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=110G
#SBATCH --time=6:00:00
#SBATCH --output=log/rpa_%j.out

eval "$(/home/carlo/tools/bin/micromamba shell hook --shell bash)"
micromamba activate triqs_env

export PMIX_MCA_psec=native
srun --mpi=pmix python mpi_scaling.py