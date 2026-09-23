#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=2400:00:00
#SBATCH -a [0-17]%4
#SBATCH -J validate-data
#SBATCH -o log/test-atom-%A-%a.log -e log/test-atom-%A-%a.err

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${ROOT_DIR}"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/test_job.sh"

export load_model_args="--load atom-2647768 --load_epoch 1265"

export if_continue_args=1
export IF_GRAD=0
export name_mol_reverse=0
export DATASET="dft-fitset-def2"

export basis_args="def2-QZVP(D)"
# export basis_args="def2-QZVPPD"

select_molecule_profile intermolecular

setup_gpu_test_job 32000 15000 1
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB); enable Numba threads.

begin_test_job
run_test_job --max_cycle 100
finish_test_job
