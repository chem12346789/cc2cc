#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2400:00:00
#SBATCH -a [12]%1
#SBATCH -J validate-full

SCRIPT_DIR="scripts"
source "${SCRIPT_DIR}/__lib__/runtime.sh"
source "${SCRIPT_DIR}/__lib__/test_job.sh"

case "${MODEL_INDEX:-0}" in
    (0) export load_model_args="--load atom-2065162 --load_epoch 2395" ;;
    # (1) export load_model_args="--load atom-2065162 --load_epoch 4955" ;;
    (1) export load_model_args="--load atom-2065162 --load_epoch 10075" ;;
esac

export if_continue_args=1
export IF_GRAD=0
export name_mol_reverse=0
export DATASET="gmtkn-def2"
export TEST_MODE="${TEST_MODE:-scf}"
export DFT2CC_MAX_GPU_JK_NAO=3500
export DFT2CC_PROFILE_XC=0

export basis_args="def2-QZVP-GMTKN55"
select_molecule_profile gmtkn55
if [[ "${name_mol_input}" == "molecule_BH76RC" ]]; then
    echo "BH76RC uses the molecules evaluated by the BH76 task."
    exit 0
fi

setup_gpu_test_job 256000 40000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).
echo "NUMBER_OF_THREADS=${NUMBER_OF_THREADS}"

find "${PYSCF_TMPDIR}" -type f -mtime +4 -exec rm -f {} \;
find "${PYSCF_TMPDIR}" -type d -empty -delete

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
test_mode="${TEST_MODE}"
if [[ "${load_model_args:-}" =~ --test_mode[[:space:]=]+([^[:space:]]+) ]]; then
    test_mode="${BASH_REMATCH[1]}"
fi
if [[ "${test_mode}" == "post-b3lyp-d3" ]]; then
    run_test_job --max_memory_gpu 1000 --max_cycle 50
else
    run_test_job --max_memory_gpu 1000 --max_cycle 50 --s6 1.0 --a1 0.8 --s8 -1.0725 --a2 0.15 --alp 14.0 --s9 1.0
fi
finish_test_job
