#!/bin/bash

#slurm options
#SBATCH -n 20
#SBATCH -p gpu
#SBATCH -J validate-data
#SBATCH -o log/data.log
#SBATCH --exclude=gpu[01-03,05-07]

## set environment variables
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_TMPDIR=~/workdir/tmp
export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/pyscf/lib:$LD_LIBRARY_PATH
export DATA_PATH=~/workdir/cadft/data/grids_mrks

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

## user's own commands below
~/anaconda3/envs/pyscf/bin/python test.py -dl -1.0 2.5 36 -b cc-pCVTZ --extend_atom 0 2 0-1 0-2 0-3 0.2-1.3 --extend_xyz 0 --load data --name_mol methane ethane ethylene acetylene cyclopropene cyclopropane propane propylene propyne allene --from_data True --require_grad True >log/data.out
#
# ~/anaconda3/envs/pyscf/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol hexane 2-methylpentane cyclohexane 1-hexene methylcyclopentane propylcyclopropane bicyclo-hexene --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float32 --load_epoch -1 >log/CHECKPOINT.out
