#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2400:00:00
#SBATCH -a [0-1]%2
#SBATCH -J benchmark_tmc_skala
#SBATCH -o log/benchmark_tmc_skala_%A_%a.out

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/test_job.sh"

# export load_model_args="--load atom-82794 --load_epoch 10005"
export load_model_args="--load test"
# export load_model_args="--load atom-590022 --load_epoch 4940" # 2395 4940
# export load_model_args="--load atom-4006776 --load_epoch 2365" # 2365 4800

export if_continue_args=0
export IF_GRAD=0
export name_mol_reverse=0
export basis_args="def2-QZVP(D)"
# export basis_args="def2-SVP"
export DATASET="tmc-def2"
select_molecule_profile tmc
# export DATASET="gmtkn-def2"
# export name_mol_input_list=("W4_11-ch" "W4_11-ch4")

setup_gpu_test_job 20000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).
export PYTHONUNBUFFERED=1
export SKALA_PYTHON_BIN="${SKALA_PYTHON_BIN:-$HOME/anaconda3/envs/skala-py3.11/bin/python}"
export LD_LIBRARY_PATH="${SKALA_LIBRARY_PATH:-$HOME/anaconda3/envs/skala-py3.11/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_external_benchmark_job "${SKALA_PYTHON_BIN}" test_skala.py --basis "${basis_args}" --dataset "${DATASET}" --name_mol "${name_mol_input}"
finish_test_job
