#!/bin/bash

# export MODEL="--model densenet --load atom-1-1916450 --load_epoch -24000"
export MODEL="--model transformer"

# export DATASET="--dataset g2"
export DATASET="--dataset gmtkn"

export dl_args="0 0 1"
# export basis_args="Def2-SVP"
export basis_args="cc-pVDZ"
export n_rad_args=""
export n_ang_args=""
export ITERS_TO_ACCUMULATE=5
export MAX_NORM=2.5

export NUMBER_OF_GPU=1
export NUMBER_OF_THREADS=16

export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export LD_PRELOAD=~/.local/lib/libjemalloc.so:$LD_PRELOAD

export PID_THIS_RUN=$$

export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_CUBE_USE=3
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

for train_atom in -1; do
	# for train_atom in 1 4 5 6 7 8 9 13 14 15 16 17; do
	export train_atom=${train_atom}
	nohup bash <<'EOF' >log/train-${PID_THIS_RUN}-${train_atom}.log 2>&1 &
set -e  # Exit on any error
export load_args="${MODEL} ${DATASET}"
echo "${mol_args}"
echo "Training atom ${train_atom}"
echo "${load_args}"
echo "Model: ${MODEL}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
if [ "$DATASET" = "--dataset gmtkn" ]; then
	export if_load_to_gpu_once=0
else
	export if_load_to_gpu_once=1
fi
~/anaconda3/envs/pyscf/bin/python train.py ${mol_args} --eval_step 5 --epoch 25010 --with_eval 0 --precision float64 ${load_args} --save_dir atom${train_atom}-${PID_THIS_RUN} --loss_multiplier 1e-2 --lr 1e-4 --train_atom ${train_atom} --iters_to_accumulate ${ITERS_TO_ACCUMULATE} --max_norm ${MAX_NORM} --if_load_to_gpu_once ${if_load_to_gpu_once} --batch_size 1 || exit 1
echo DONE
EOF
done

echo $! >>log/save_pid.txt 2>&1
