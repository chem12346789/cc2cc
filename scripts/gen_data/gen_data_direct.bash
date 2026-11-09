#!/bin/bash
#SBATCH --array=[0-3]%4
# CLUSTER-ONLY: This script is configured for this cluster; do not run it elsewhere.

if [[ "$(hostname)" == "user-MZ73-LM1-000" ]]; then
    :
else
    printf 'Error: this script can only run on user-MZ73-LM1-000 (current host: %s).\n' "$(hostname)" >&2
    exit 1
fi

# Generation settings
# export GEN_CONFIG="gen_mol_addon.json"
# export GEN_CONFIG="gen_mol2.json"
export GEN_CONFIG="gen_error.json"
export CONVERGENCE=0

export DATASET="gmtkn-def2"
export basis_args="def2-QZVPPD"
export MD_NUMBER=0
export IF_CONTINUE=1

# Slurm array task
export MP_TOTAL="${SLURM_ARRAY_TASK_COUNT:?Submit this script as a Slurm array}"
export MP_NUMBER="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

# Cluster runtime
export PID_THIS_RUN=$$
export LD_PRELOAD=~/.local/lib/libjemalloc.so:$LD_PRELOAD
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/pyscf/lib:$LD_LIBRARY_PATH
export DFT2CC_CUBE_USE=3
export PYSCF_TMPDIR=~/raid/tmp

# Resource limits
export NUMBER_OF_THREADS=16
export PYSCF_MAX_MEMORY=120000
export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}
export VECLIB_MAXIMUM_THREADS=${NUMBER_OF_THREADS}
export BLIS_NUM_THREADS=${NUMBER_OF_THREADS}
export OMP_WAIT_POLICY=PASSIVE

# Output directories
mkdir -p log
mkdir -p data/grids_dft

# Run generation
log_file="log/gen_data-${PID_THIS_RUN}-${MP_NUMBER}.log"
err_file="log/gen_data-${PID_THIS_RUN}-${MP_NUMBER}.err"
if [[ "${DIRECT_RUN_LOG_LINKS:-}" == "1" ]]; then
    printf '[TASK-LOG] model=%s task=%s log=%s err=%s\n' \
        "${MODEL_INDEX:-0}" "${MP_NUMBER}" "${PWD}/${log_file}" "${PWD}/${err_file}"
fi
exec ~/anaconda3/envs/pyscf/bin/python gen_data.py \
    --basis "${basis_args}" --dataset "${DATASET}" --if_continue "${IF_CONTINUE}" \
    --if_eval 0 --name_mol_reverse 0 --md_number "${MD_NUMBER}" --device cpu \
    --mp_number "${MP_NUMBER}" --mp_total "${MP_TOTAL}" --grid_level 4 \
    --zmp_feature_ov false --zmp_observable_response false \
    --zmp_constraint_metadata true \
    --gen_config "${GEN_CONFIG}" \
    --check_convergence "${CONVERGENCE}" \
    >"${log_file}" \
    2>"${err_file}"
