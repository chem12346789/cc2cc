#!/bin/bash

#slurm options
#SBATCH -n 24
#SBATCH --mem 100000
#SBATCH --nodelist=gpu06
#SBATCH -p gpu
#SBATCH -J gen_data_MOL_EXTEND_ATOM
#SBATCH -o log/%j.log

## user's own commands below
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMBA_NUM_THREADS=24

export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_DATA_PATH=~/workspace/cc2cc/data/grids_dft_mix/

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

export DFT2CC_CUBE_USE=3

nohup bash -c '~/anaconda3/envs/pyscf/bin/python krr_fit.py --alpha 1e-8 --gamma 1000 --distance_list -0.5 0.5 3 --molecular_list methane ethane ethylene acetylene' >log/krr5.log 2>&1 &
# nohup bash -c '~/anaconda3/envs/pyscf/bin/python krr_fit.py --alpha 1e-8 --gamma 1000 --distance_list -0.5 0.5 11 --molecular_list methane' >log/krr2.log 2>&1 &
echo $! >>log/save_pid.txt 2>&1
