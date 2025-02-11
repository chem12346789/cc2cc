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

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

# basis_args="Def2-TZVPD"
# nohup bash -c "~/anaconda3/envs/pyscf/bin/python test.py -dl 0 0 1 --basis ${basis_args} --extend_atom 0 --extend_xyz 0 --precision float64 --load cycle1 --load_epoch 15000 --dataset g2" >log/test-${basis_args}.log 2>&1 &

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft

basis_args="cc-pVDZ"

nohup bash -c "~/anaconda3/envs/pyscf/bin/python test.py -dl 0 0 1 --basis ${basis_args} --extend_atom 0 --extend_xyz 0 --precision float64 --load cycle1 --load_epoch 15000 --dataset g2" >log/test-${basis_args}.log 2>&1 &

echo $! >>log/save_pid.txt 2>&1
