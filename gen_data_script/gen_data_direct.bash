#!/bin/bash

# Parameters for train.py
export dl_args="-0.5 0.5 3"
# export basis_args="Def2-SVP"
export basis_args="cc-pVDZ"
export n_rad_args=""
export n_ang_args=""

## user's own commands below
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

export PID_THIS_RUN=$$

export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_CUBE_USE=3
export DFT2CC_GENERATE_DATA=1
export PYSCF_TMPDIR=~/raid/tmp

export NVIDIA_VISIBLE_DEVICES=1
# use less power GPU
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# use most free memory GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -n | tail -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft

if [ -z "$n_rad_args" ]; then
	export mol_args="--distance_list ${dl_args} --basis ${basis_args} --extend_atom 0-1 --extend_xyz 0"
else
	export mol_args="--distance_list ${dl_args} --basis ${basis_args} --n_rad ${n_rad_args} --n_ang ${n_ang_args} --extend_atom 0-1 --extend_xyz 0"
fi

nohup bash <<'EOF' >log/gen_data-${PID_THIS_RUN}.log 2>&1 &
set -e  # Exit on any error
~/anaconda3/envs/pyscf/bin/python gen_data.py ${mol_args} --name_mol molecule0 molecule1 molecule2 molecule3 molecule4 molecule5 --dataset g2-ase || exit 1
echo DONE
EOF

echo $! >>log/save_pid.txt 2>&1
