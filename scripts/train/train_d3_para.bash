#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0%1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=20:00:00
#SBATCH -J train-d3-para
#SBATCH -o log/train-d3-para-%A.log -e log/train-d3-para-%A.err
#SBATCH --exclude=gpu[01,02,03,04,05,07]

set -eo pipefail

export basis_args="def2-QZVP(D)"
case "${MODEL_INDEX:-0}" in
    (0) export load_model_args="--load atom-3259973 --load_epoch 2395" ;;
esac

if [[ "${load_model_args:-}" =~ --load[[:space:]]+([^[:space:]]+) ]]; then
    export load_args="${BASH_REMATCH[1]}"
else
    printf 'Missing --load MODEL in load_model_args: %s\n' "${load_model_args:-}" >&2
    exit 2
fi

if [[ "${load_model_args:-}" =~ --load_epoch[[:space:]]+([0-9]+)($|[[:space:]]) ]]; then
    load_epoch="${BASH_REMATCH[1]}"
else
    printf 'Missing --load_epoch EPOCH in load_model_args: %s\n' "${load_model_args:-}" >&2
    exit 2
fi

if [[ -n "${FORCE_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$FORCE_CUDA_VISIBLE_DEVICES"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,power.draw,index --format=csv,nounits,noheader | awk -F, '$1 > 2000 {print $2, $3}' | sort -n | head -1 | awk '{print $NF}')
    if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
        printf 'No GPU has more than 2000 MiB free memory.\n' >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES
fi

if [[ "${SLURM_ARRAY_TASK_ID:-0}" != 0 ]]; then
    printf 'The D3 sweep uses one task per model; array task must be 0.\n' >&2
    exit 2
fi

export VALIDATE_DIR="validate_hkqai"

export NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-10}
export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}
export LD_PRELOAD=~/.local/lib/libjemalloc.so:$LD_PRELOAD

export PYSCF_MAX_MEMORY=8000
export PYTHON_BIN=~/anaconda3/envs/pyscf-cuda12/bin/python
export LOSS_TYPE="mse"

for dataset in dft-fitset-def2 gmtkn-diet100-def2; do
    data_file="${VALIDATE_DIR}/ccdft_${basis_args}_${load_args}_${load_epoch}_${dataset}.csv"
    if [[ ! -r "$data_file" ]]; then
        printf 'Missing D3 input CSV: %s\n' "$data_file" >&2
        exit 1
    fi
done

mkdir -p log validate
run_timestamp=$(date +%Y%m%d-%H%M%S)
run_dir="log/d3-sweep_${load_args}_epoch-${load_epoch}_${basis_args}_bj_${LOSS_TYPE}_${run_timestamp}_pid-$$"
mkdir -- "$run_dir"
printf 'D3 sweep outputs: %s\n' "$run_dir"

mapfile -t rs18_values < <(seq -1.0 0.1 0.2)
mapfile -t rs6_values < <(seq 0.5 0.1 1.5)

for rs18 in "${rs18_values[@]}"; do
    for rs6 in "${rs6_values[@]}"; do
        echo "Training with rs6=${rs6}, rs18=${rs18}..."
        point_prefix="${run_dir}/rs6-${rs6}_rs18-${rs18}"

        "${PYTHON_BIN}" d3_para.py \
            --mode train \
            --basis "${basis_args}" \
            --load "${load_args}" \
            --load_epoch "${load_epoch}" \
            --epochs 1000 \
            --print_step 1 \
            --lr 1e-2 \
            --dataset gmtkn-def2 \
            --damping bj \
            --loss-type "${LOSS_TYPE}" \
            --s6 1.0 \
            --alp 14 \
            --rs6 "${rs6}" \
            --rs18 "${rs18}" \
            --optimizer lbfgs 2>&1 | tee "${point_prefix}-train.log"

        cp -- "${VALIDATE_DIR}/ccdft_${basis_args}_${load_args}_${load_epoch}_bj_dft-fitset-def2.json" \
            "${point_prefix}.json"

        "${PYTHON_BIN}" d3_para.py \
            --mode test \
            --basis "${basis_args}" \
            --load "${load_args}" \
            --load_epoch "${load_epoch}" \
            --dataset gmtkn-diet100-def2 \
            --damping bj 2>&1 | tee "${point_prefix}-test.log"

        MPLBACKEND=Agg "${PYTHON_BIN}" scripts/plot_d3_sweep.py \
            "$run_dir" "$rs6" "$rs18" "$point_prefix" \
            --rs6-values "${rs6_values[@]}" \
            --rs18-values "${rs18_values[@]}"
    done
done
