#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2400:00:00
#SBATCH -a [0]%1
#SBATCH -J validate-force
#SBATCH -o log/test-force-%A.log -e log/test-force-%A.err

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/test_job.sh"

export load_model_args="--load atom-1705553 --load_epoch 1115"
# export load_model_args="--load atom-3054023 --load_epoch 475"
# export load_model_args="--load atom-3055766 --load_epoch 1115"
# export load_model_args="--load atom-1692437 --load_epoch 2380"
# export load_model_args="--load atom-82794 --load_epoch 10005"

export if_continue_args=0
export name_mol_reverse=0
export IF_GRAD=1
export DATASET="gmtkn-def2"

export basis_args="def2-QZVPPD"
export name_mol_input="W4_11-ch4"

# export basis_args="def2-TZVP"
# export name_mol_input="molecule_Force"

setup_gpu_test_job 16000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_cycle 50
finish_test_job
