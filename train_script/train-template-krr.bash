#!/bin/bash

#slurm options
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem 50G
#SBATCH -p cpu
#SBATCH -J train-ccdft-BASH_GAMMA-BASH_ALPHA
#SBATCH -o log/BASH_GAMMA-BASH_ALPHA.log

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/envs/sklearn/bin/python krr.py --gamma BASH_GAMMA --alpha BASH_ALPHA
