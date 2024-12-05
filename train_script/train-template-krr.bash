#!/bin/bash

#slurm options
#SBATCH -n 24
#SBATCH -p gpu
#SBATCH -J train-ccdft-BASH_GAMMA-BASH_ALPHA
#SBATCH -o log/BASH_GAMMA-BASH_ALPHA.log

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMBA_NUM_THREADS=12

export DFT2CC_CUBE_USE=1
export DFT2CC_PERIOD=1
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/envs/pyscf/bin/python krr_view.py --gamma BASH_GAMMA --alpha BASH_ALPHA --distance_list -0.5 0.5 11 --molecular_list methane ethane ethylene acetylene propane
# ~/anaconda3/envs/pyscf/bin/python krr_view.py --gamma BASH_GAMMA --alpha BASH_ALPHA --distance_list 0 0 1 --molecular_list methane
