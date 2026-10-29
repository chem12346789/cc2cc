#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2400:00:00
#SBATCH -a [0]%1
#SBATCH -J validate-rotate

SCRIPT_DIR="scripts"
source "${SCRIPT_DIR}/__lib__/runtime.sh"
source "${SCRIPT_DIR}/__lib__/test_job.sh"

case "${MODEL_INDEX:-0}" in
    # (0) export load_model_args="--load atom-393567 --load_epoch 1115" ;;
    (0) export load_model_args="--load atom-35580 --load_epoch 2395" ;;
esac

export if_continue_args=0
export name_mol_reverse=0
export IF_GRAD="${IF_GRAD:-0}"
export GRID_RESPONSE="${GRID_RESPONSE:-0}"
export DATASET="rotation-def2"

export basis_args="def2-QZVPPD"
select_molecule_profile rotate
export name_mol_input="${ROTATION_SET:-${name_mol_input}}"

setup_gpu_test_job 256000 16000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_memory_gpu 8000 --max_cycle 50
finish_test_job
