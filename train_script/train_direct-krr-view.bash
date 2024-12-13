#!/bin/bash

#slurm options
#SBATCH -n 24
#SBATCH --mem 100000
#SBATCH --nodelist=gpu06
#SBATCH -p gpu
#SBATCH -J gen_data_MOL_EXTEND_ATOM
#SBATCH -o log/%j.log

## user's own commands below
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/home/chenzihao/anaconda3/envs/pyscf/lib:$LD_LIBRARY_PATH
export DFT2CC_DATA_PATH=~/workspace/cc2cc/data/grids_dft_mix/

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU
export DFT2CC_CUBE_USE=3

# nohup bash -c '~/anaconda3/envs/pyscf/bin/python krr_view.py --alpha 1e-8 --gamma 1000 0.1 0.1 --distance_list 0 0 1 --molecular_list methane ethane ethylene acetylene propane' >log/view.out &
# nohup bash -c '~/anaconda3/envs/pyscf/bin/python krr_view.py --alpha 1e-8 --gamma 1000  --distance_list -0.5 0.5 3 --molecular_list methane ethane ethylene acetylene propane' >log/view2.out 2>&1 &
nohup bash -c '~/anaconda3/envs/pyscf/bin/python krr_view.py --alpha 1e-8 --gamma 1000  --distance_list 0 0 1 --molecular_list methane' >log/view2.out 2>&1 &

echo $! >save_pid.txt
