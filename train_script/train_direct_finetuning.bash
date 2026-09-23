#!/bin/bash

# =========================
# Model and checkpoint
# =========================
export MODEL="--model transformer+dense_mix_e3nn_4"
# export MODEL="--model transformer+dense_mix_e3nn_7"
export IF_RESUME=0
export LOAD_MODEL_ARGS="--if_resume ${IF_RESUME}"
# export LOSS_TYPE_ARGS="--loss_type L1Loss"

# =========================
# Data and targets
# =========================
export DATASET="gmtkn-def2"
export BASIS_ARGS="def2-QZVPPD"

# Options: dft, dft_d3bj, zmp
# export RHO_INPUT="zmp"
export RHO_INPUT="dft"
# export RHO_INPUT="dft_d3bj"
# export SPLIT_CONFIG="mol1.json"
# export SPLIT_CONFIG="mol2.json"
export SPLIT_CONFIG="test.json"

export OUTPUT_ARG="--output_target tol_delta_grids"
# export OUTPUT_ARG="--output_target b3lyp"

# =========================
# Loss terms
# =========================
export IF_GRAD=0
export IF_ATOMIC=0
export IF_ABS=0

export GRAD_ARG="--loss_multiplier_grad 1 --if_grad ${IF_GRAD}"
export ATOMIC_ARG="--loss_multiplier_atomic 1 --if_atomic ${IF_ATOMIC}"
export ABS_ARG="--loss_multiplier_abs 1 --if_relative_weight_abs 1 --if_abs ${IF_ABS}"
# export ABS_ARG="--loss_multiplier_abs 1 --if_abs ${IF_ABS}"

# =========================
# Optimizer and scheduler
# =========================
export LEARNING_RATE="1e-4"
export COSINE_ETA_MIN_RATIO="1e-3"
export COSINE_ETA_MIN_EFFECTIVE=$(awk -v lr="${LEARNING_RATE}" -v ratio="${COSINE_ETA_MIN_RATIO}" 'BEGIN { printf "%.12g", lr * ratio }')

export WEIGHT_DECAY="1e-12"
export MAX_NORM=1
export SCHEDULER="--optimizer AdamW --scheduler cosine_warm --cosine_eta_min ${COSINE_ETA_MIN_EFFECTIVE} --cosine_T 32"
# export SCHEDULER="--optimizer Adafactor --scheduler cosine_warm --cosine_eta_min ${COSINE_ETA_MIN_EFFECTIVE}"

# =========================
# Hardware and environment
# =========================
export NUMBER_OF_GPU=1
export CUDA_VISIBLE_DEVICES=5

if [ ${NUMBER_OF_GPU} -gt 1 ]; then
	export DISTRIBUTED="--distributed 1"
else
	export DISTRIBUTED="--distributed 0"
fi

export THREADS_PER_GPU=1
export NUMBER_OF_THREADS=$((NUMBER_OF_GPU * THREADS_PER_GPU))

export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}

export PYTHONPATH=$HOME/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/pyscf-cuda12/lib:$LD_LIBRARY_PATH
export PYTHON_PATH=$HOME/anaconda3/envs/pyscf-cuda12/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=$HOME/tmp
export TORCHRUN_PATH="$HOME/anaconda3/envs/pyscf-cuda12/bin/torchrun \
	--standalone \
	--nnodes=1 \
	--nproc_per_node=${NUMBER_OF_GPU} \
	--master_port=61235"

export LOG_DIR="log"
mkdir -p "${LOG_DIR}"

train_run() {
	set -e
	echo "Starting training run with PID ${PID_THIS_RUN}..."
	echo "Sleeping for 5 hours to wait for the previous run to finish..."
	sleep 5h
	kill 3981648

	local train_args=(
		${DISTRIBUTED}
		${MODEL}
		${LOAD_MODEL_ARGS}
		--save_dir "atom-${PID_THIS_RUN}"
		--eval_step 5
		--epoch 250001
		--max_norm "${MAX_NORM}"
		--seed 42
		--rho_input "${RHO_INPUT}"
		${OUTPUT_ARG}
		--precision float64
		--lr "${LEARNING_RATE}"
		--weight_decay "${WEIGHT_DECAY}"
		${SCHEDULER}
		--dataset "${DATASET}"
		--basis "${BASIS_ARGS}"
		--split_config "${SPLIT_CONFIG}"
		--atomic_weighting 2
		--if_relative_weight 1
		${LOSS_TYPE_ARGS}
		${GRAD_ARG}
		${ATOMIC_ARG}
		${ABS_ARG}
	)

	echo "Training run ${PID_THIS_RUN}"
	echo "Model: ${MODEL}"
	echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
	echo "LR: ${LEARNING_RATE}"
	echo "Cosine eta_min ratio (--cosine_eta_min): ${COSINE_ETA_MIN_RATIO}"
	echo "Cosine eta_min effective absolute LR (after parser scaling): ${COSINE_ETA_MIN_EFFECTIVE}"
	echo "Weight decay: ${WEIGHT_DECAY}"
	echo "Scheduler: ${SCHEDULER}"
	echo "Split config: ${SPLIT_CONFIG}"

	OMP_WAIT_POLICY=PASSIVE ${TORCHRUN_PATH} train.py "${train_args[@]}"
	echo DONE
}
export -f train_run

export PID_THIS_RUN=$$
echo "Starting train script with PID ${PID_THIS_RUN}..."
nohup bash -c "train_run" >"${LOG_DIR}/train-${PID_THIS_RUN}.log" 2>&1 &
echo $! >>"${LOG_DIR}/save_pid.txt"
