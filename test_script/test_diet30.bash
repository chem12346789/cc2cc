#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2400:00:00
#SBATCH -a [0-54]%2
#SBATCH -J validate-diet30
###SBATCH --exclude=gpu[01-03,05-07]

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${ROOT_DIR}/test_script"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/test_job.sh"

# export load_model_args="--load atom-2647768 --load_epoch 1265"
export load_model_args="--load atom-2359049 --load_epoch 2395"
# export load_model_args="--load atom-3232520 --load_epoch 2395"
# export load_model_args="--load atom-3354597 --load_epoch 2395"
# export load_model_args="--load atom-2359049 --load_epoch 1115"

export if_continue_args=0
export IF_GRAD=0
export name_mol_reverse=0
export DATASET="gmtkn-diet30-def2"

export basis_args="def2-QZVP(D)"
select_molecule_profile gmtkn55

setup_gpu_test_job 16000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_cycle 50 --s6 1.0 --a1 0.8 --s8 -1.0725 --a2 0.15 --alp 14.0 --s9 1.0
finish_test_job
