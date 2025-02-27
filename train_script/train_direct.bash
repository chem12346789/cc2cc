#!/bin/bash

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PID_THIS_RUN=$$

export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_CUBE_USE=3
export DFT2CC_GENERATE_DATA=1
export PYSCF_TMPDIR=~/raid/tmp

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft

# export dl_args="-0.5 0.5 11"
export dl_args="0 0 1"
# export basis_args="Def2-TZVPD"
export basis_args="cc-pVDZ"
export n_rad_args="302"
export n_ang_args="302"
if [ -z "$n_rad_args" ]; then
	export mol_args="--distance_list ${dl_args} --basis ${basis_args} --extend_atom 0-1 --extend_xyz 0"
else
	export mol_args="--distance_list ${dl_args} --basis ${basis_args} --n_rad ${n_rad_args} --n_ang ${n_ang_args} --extend_atom 0-1 --extend_xyz 0"
fi

nohup bash <<'EOF' >log/train-${PID_THIS_RUN}.log 2>&1 &
set -e  # Exit on any error
# ~/anaconda3/envs/pyscf/bin/python gen_data.py ${mol_args} --dataset g2 || exit 1
for train_atom in 1; do
# for train_atom in 1 4 5 6 7 8 9 13 14 15 16 17; do
	# export load_args="--load atom-1-1025406"
	export load_args=""
	echo "${mol_args}"
	echo "Training atom ${train_atom}"
	echo "${load_args}"
	~/anaconda3/envs/pyscf/bin/python train.py ${mol_args} --eval_step 5 --epoch 25100 --with_eval 0 --precision float32 ${load_args} --save_dir atom${train_atom}-${PID_THIS_RUN} --loss_multiplier 0.01 --lr 1e-4 --train_atom ${train_atom} || exit 1
done
echo DONE
EOF

echo $! >>log/save_pid.txt 2>&1
