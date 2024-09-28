#!/bin/bash

#slurm options
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --nodelist=gpu06
#SBATCH -J train-ccdft-BASH_EVAL_STEP-BASH_BATCH_SIZE-BASH_WITH_EVAL_BASH_STRUCTURE
#SBATCH -o log/%j.log

## user's own commands below
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=BASH_NUMBER_OF_GPU

export PYTHONPATH=~/python:$PYTHONPATH
export PYSCF_MAX_MEMORY=80000
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_DATA_PATH=~/workdir/dft2cc/data/grids_dft_3_0.005/
export DFT2CC_STRUCTURE=BASH_STRUCTURE
export DFT2CC_CUBE_USE=BASH_CUBE_USE
# export DFT2CC_TEST=True

~/anaconda3/envs/pyscf/bin/python train.py -dl -0.5 0.5 21 -b cc-pVDZ --extend_atom 0 0-1 --extend_xyz 0 --eval_step BASH_EVAL_STEP --batch_size BASH_BATCH_SIZE --epoch 2500 --with_eval BASH_WITH_EVAL --precision float32 --load BASH_LOAD_MODEL
# ~/anaconda3/envs/pyscf/bin/python train.py -dl 0 0 1 -b cc-pVDZ --extend_atom 0 --extend_xyz 0 --eval_step BASH_EVAL_STEP --batch_size BASH_BATCH_SIZE --epoch 2500 --with_eval BASH_WITH_EVAL --precision float32 --load BASH_LOAD_MODEL
