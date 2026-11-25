#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=2400:00:00
#SBATCH -a [0-54]%1
#SBATCH -J validate-data
#SBATCH -o log/detail/benchmark-%A-%a.log -e log/detail/benchmark-%A-%a.err

SCRIPT_DIR="scripts"
source "${SCRIPT_DIR}/__lib__/runtime.sh"
source "${SCRIPT_DIR}/__lib__/test_job.sh"

export load_epoch="-1"
export IF_GRAD=0

export if_continue_args=0
export name_mol_reverse=0
export basis_args="def2-QZVP-GMTKN55"
export DATASET="gmtkn-def2"
select_molecule_profile gmtkn55
if [[ "${name_mol_input}" == "molecule_BH76RC" ]]; then
    echo "BH76RC uses the molecules evaluated by the BH76 task."
    exit 0
fi
setup_gpu_test_job 250000 20000
begin_test_job benchmark_dft
run_gpu_benchmark_job
finish_test_job