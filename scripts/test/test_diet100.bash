#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=2400:00:00
#SBATCH -a [0-54]%4
#SBATCH -J validate-diet100

SCRIPT_DIR="scripts"
source "${SCRIPT_DIR}/__lib__/runtime.sh"
source "${SCRIPT_DIR}/__lib__/test_job.sh"

case "${MODEL_INDEX:-0}" in
0) export load_model_args="--load atom-1563701 --load_epoch 4955" ;;
1) export load_model_args="--load atom-1221273 --load_epoch 2395" ;;
2) export load_model_args="--load atom-1560444 --load_epoch 4955" ;;
3) export load_model_args="--load atom-82794 --load_epoch 4940" ;;
4) export load_model_args="--load atom-82794 --load_epoch 2395" ;;
5) export load_model_args="--load atom-82794 --load_epoch 1115" ;;
esac

export if_continue_args=1
export IF_GRAD=0
export name_mol_reverse=0
export DATASET="gmtkn-diet100-def2"

export basis_args="def2-QZVP(D)"
select_molecule_profile gmtkn55

setup_gpu_test_job 64000 15000
# Arguments: PySCF memory (MiB); minimum free GPU memory (MiB).
echo "NUMBER_OF_THREADS=${NUMBER_OF_THREADS}"

find "${PYSCF_TMPDIR}" -type f -mtime +4 -exec rm -f {} \;
find "${PYSCF_TMPDIR}" -type d -empty -delete

sleep $SLURM_ARRAY_TASK_ID
begin_test_job
run_test_job --max_cycle 50 --s6 1.0 --a1 0.8 --s8 -1.0725 --a2 0.15 --alp 14.0 --s9 1.0
finish_test_job
