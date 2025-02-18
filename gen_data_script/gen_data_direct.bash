#!/bin/bash

#slurm options
#SBATCH -n 24
#SBATCH --mem 100000
#SBATCH --nodelist=gpu06
#SBATCH -p gpu
#SBATCH -J gen_data_MOL_EXTEND_ATOM
#SBATCH -o log/%j.log

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_DATA_PATH=~/workspace/cc2cc/data/grids_dft_mix/

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft

nohup bash -c '~/anaconda3/envs/pyscf/bin/python gen_data.py -dl -0.5 0.5 5 --basis cc-pVDZ --extend_atom 0-1 --extend_xyz 0 --name_mol g2 --precision float64' >log/gen_data_g2.log 2>&1 &
# nohup bash -c '~/anaconda3/envs/pyscf/bin/python gen_data.py -dl -0.5 0.5 11 --basis cc-pVDZ --extend_atom 0-1 --extend_xyz 0 --name_mol g2 --precision float64'  > log/gen_data_MOL.log 2>&1 &

echo $! >>log/save_pid.txt 2>&1
