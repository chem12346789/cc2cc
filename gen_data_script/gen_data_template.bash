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

export PYSCF_TMPDIR=~/workdir/tmp
export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_DATA_PATH=~/workdir/dft2cc/data/grids_dft_3_0.005/

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

# ~/anaconda3/envs/pyscf/bin/python gen_data.py -dl START END STEP -b BASIS --extend_atom EXTEND_ATOM --extend_xyz 0 --name_mol hexane
~/anaconda3/envs/pyscf/bin/python gen_data.py -dl START END STEP -b BASIS --extend_atom EXTEND_ATOM --extend_xyz 0 --name_mol methane ethane ethylene acetylene cyclopropene cyclopropane allene propyne propane propylene butane butyne isobutane butadiene bicyclobutane cyclobutane benzene spiropentane cyclopropylmethyl neopentane cyclopentane pentane isopentane
# 
# ~/anaconda3/envs/pyscf/bin/python gen_data.py -dl START END STEP -b BASIS --extend_atom EXTEND_ATOM --extend_xyz 0 --name_mol spiropentane cyclopropylmethyl neopentane cyclopentane pentane isopentane
