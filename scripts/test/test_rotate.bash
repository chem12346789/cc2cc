#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2400:00:00
#SBATCH -J test-rotate
#SBATCH -o log/test-rotate-%A.log -e log/test-rotate-%A.err

SCRIPT_DIR="scripts"
source "${SCRIPT_DIR}/__lib__/runtime.sh"
source "${SCRIPT_DIR}/__lib__/test_job.sh"

export load_model_args="--load atom-82794 --load_epoch 10005"
# export name_mol_input="W4_11-b W4_11-o HEAVY28-bih3"
export name_mol_input="W4_11-ch4 Force-ch4-1.000"

export if_continue_args=0
export name_mol_reverse=0
export IF_GRAD=0
export DATASET="gmtkn-def2"

# export basis_args="def2-TZVP(D)"
export basis_args="def2-QZVP(D)"
# export basis_args="def2-TZVPPD"
# export basis_args="def2-QZVPPD"

setup_gpu_test_job 16000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_cycle 50
finish_test_job
