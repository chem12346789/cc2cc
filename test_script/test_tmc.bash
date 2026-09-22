#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2400:00:00
#SBATCH -a [0-1]%2
#SBATCH -J validate-diet30

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/test_job.sh"

# export load_model_args="--load atom-571426 --load_epoch " #
# export load_model_args="--load atom-1317610 --load_epoch " #
export load_model_args="--load atom-590022 --load_epoch 10005" # 2395 4940 10005
# export load_model_args="--load atom-82794 --load_epoch 10005"

export if_continue_args=1
export IF_GRAD=0
export name_mol_reverse=0
export basis_args="def2-QZVP(D)"
export DATASET="tmc-def2"
select_molecule_profile tmc

setup_gpu_test_job 8000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_cycle 50
finish_test_job
