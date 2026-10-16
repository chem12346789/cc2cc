#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2400:00:00
#SBATCH -a [0]%1
#SBATCH -J validate-force

SCRIPT_DIR="scripts"
source "${SCRIPT_DIR}/__lib__/runtime.sh"
source "${SCRIPT_DIR}/__lib__/test_job.sh"

case "${MODEL_INDEX:-0}" in
0) export load_model_args="--load atom-7297 --load_epoch 1115" ;;
esac

export if_continue_args=0
export name_mol_reverse=0
export IF_GRAD=1
export GRID_RESPONSE="${GRID_RESPONSE:-1}"
export DATASET="gmtkn-def2"

export basis_args="def2-QZVPPD"
select_molecule_profile force

# export basis_args="def2-TZVP"
# export name_mol_input="molecule_Force"

setup_gpu_test_job 16000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_cycle 50
finish_test_job
