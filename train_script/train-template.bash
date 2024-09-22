#!/bin/bash

#slurm options
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --nodelist=gpu03
#SBATCH -J train-ccdft-EVAL_STEP-BATCH_SIZE-WITH_EVAL
#SBATCH -o log/%j.log

## user's own commands below
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

export PYTHONPATH=~/python:$PYTHONPATH
export PYSCF_MAX_MEMORY=80000
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DATA_PATH=~/workdir/dft2cc/data/grids_dft/

~/anaconda3/envs/pyscf/bin/python train.py -dl -0.5 0.5 11 -b cc-pVDZ --extend_atom 0 0-1 --extend_xyz 0 --eval_step EVAL_STEP --batch_size BATCH_SIZE --epoch 25000 --with_eval WITH_EVAL --precision float64 --load LOAD_MODEL
