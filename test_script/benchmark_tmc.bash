#!/bin/bash
#slurm options
#SBATCH -p cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=2400:00:00
#SBATCH -a [0-1]%2
#SBATCH -J validate-diet30

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${ROOT_DIR}/test_script"
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
setup_test_job 8000
export PYTHONUNBUFFERED=1

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_cycle 50 --device cpu --benchmark_method B3LYP --benchmark_disp None
finish_test_job
