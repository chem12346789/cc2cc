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

export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

nohup bash -c '~/anaconda3/envs/pyscf/bin/python test.py -dl -0.25 -0.25 11 --basis cc-pVDZ --extend_atom 0-1 --extend_xyz 0 --precision float64 --load 2024-12-13-13-11-44 --load_epoch 49000 --name_mol methane ethane ethylene acetylene propane' >log/test0.log 2>&1 &
echo $! >>log/save_pid.txt 2>&1
