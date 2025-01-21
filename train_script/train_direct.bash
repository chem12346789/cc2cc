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

export PID_THIS_RUN=$$

export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_MAIN_PATH=~/workspace/cc2cc_test2
export DFT2CC_CUBE_USE=3
export DFT2CC_GENERATE_DATA=1

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft
nohup bash <<'EOF' >log/train-$$.log 2>&1 &
set -e  # Exit on any error
for cycle in {2..8}; do
	prev_cycle=$((cycle-1))
	load_args=""
	dl_args="-0.2 0.2 5"
	if [ $cycle -gt 1 ]; then
		load_args="--load cycle${prev_cycle} --load_epoch -10000"
		~/anaconda3/envs/pyscf/bin/python test.py -dl ${dl_args} --basis cc-pVDZ --extend_atom 0-1 --extend_xyz 0 \
			--precision float64 ${load_args} --density_restriction 1 || exit 1
	else
		~/anaconda3/envs/pyscf/bin/python gen_data.py -dl ${dl_args} --basis cc-pVDZ --extend_atom 0-1 --extend_xyz 0 || exit 1
	fi
	~/anaconda3/envs/pyscf/bin/python train.py -dl ${dl_args} --basis cc-pVDZ --extend_atom 0-1 --extend_xyz 0 \
		--eval_step 10 --epoch 29100 --with_eval True --precision float32 --save_dir cycle${cycle} \
		--loss_multiplier 0.01 ${load_args} || exit 1
done
EOF

echo $! >>log/save_pid.txt 2>&1
