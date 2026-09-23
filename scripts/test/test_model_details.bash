#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2400:00:00
#SBATCH -J test-model-detail
#SBATCH -o log/test-model-detail-%A.log -e log/test-model-detail-%A.err

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${ROOT_DIR}"
source "${SCRIPT_DIR}/__lib__/runtime.sh"
source "${SCRIPT_DIR}/__lib__/test_job.sh"

export load_model_args="--load atom-1705553 --load_epoch 2395"
# export load_model_args="--load atom-3054023 --load_epoch 475"
# export load_model_args="--load atom-3055766 --load_epoch 1115"
# export load_model_args="--load atom-1692437 --load_epoch 2380"
# export load_model_args="--load atom-82794 --load_epoch 10005"

export if_continue_args=1
export IF_GRAD=0
export name_mol_reverse=0
export DATASET="gmtkn-diet30-def2"

export basis_args="def2-QZVPPD"

setup_gpu_test_job 16000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).

begin_test_job
run_model_detail_job --split_config mol1.json
finish_test_job
