#!/usr/bin/env bash

setup_test_job() {
    local max_memory_mb=$1
    local use_numba=${2:-0}

    export NUMBER_OF_GPU=1
    export NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
    export OMP_NUM_THREADS="${NUMBER_OF_THREADS}"
    export MKL_NUM_THREADS="${NUMBER_OF_THREADS}"
    export NUMEXPR_NUM_THREADS="${NUMBER_OF_THREADS}"
    export OPENBLAS_NUM_THREADS="${NUMBER_OF_THREADS}"
    if [[ "${use_numba}" == "1" ]]; then
        export NUMBA_NUM_THREADS="${NUMBER_OF_THREADS}"
    fi

    export PYSCF_MAX_MEMORY="${max_memory_mb}"
    export PYSCF_TMPDIR="${PYSCF_TMPDIR:-${TMPDIR:-$HOME/tmp}}"
    mkdir -p "${REPO_ROOT}/log" "${REPO_ROOT}/validate"
}

select_gpu() {
    local min_free_memory_mb=$1

    export NVIDIA_VISIBLE_DEVICES=1
    if [[ -n "${FORCE_CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${FORCE_CUDA_VISIBLE_DEVICES}"
    else
        export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=memory.free,power.draw,index --format=csv,nounits,noheader | awk -F, -v minimum="${min_free_memory_mb}" '$1 > minimum {print $2 $3}' | sort -n | head -1 | awk '{print $NF}')"
    fi
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

setup_gpu_test_job() {
    local max_memory_mb=$1
    local min_free_memory_mb=$2
    local use_numba=${3:-0}

    setup_test_job "${max_memory_mb}" "${use_numba}"
    select_gpu "${min_free_memory_mb}"
}

setup_cpu_test_job() {
    local max_memory_mb=$1

    export NUMBER_OF_GPU=0
    export NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
    export OMP_NUM_THREADS="${NUMBER_OF_THREADS}"
    export MKL_NUM_THREADS="${NUMBER_OF_THREADS}"
    export NUMEXPR_NUM_THREADS="${NUMBER_OF_THREADS}"
    export OPENBLAS_NUM_THREADS="${NUMBER_OF_THREADS}"
    export PYSCF_MAX_MEMORY="${max_memory_mb}"
    export PYSCF_TMPDIR="${PYSCF_TMPDIR:-${TMPDIR:-$HOME/tmp}}"
    mkdir -p "${REPO_ROOT}/log" "${REPO_ROOT}/validate"
}

begin_test_job() {
    local job_name=${1:-test}

    echo "Starting ${job_name} with PID $$..."
    set -e
}

finish_test_job() {
    echo "Test completed successfully."
    echo "DONE"
}

select_molecule_profile() {
    local profile=$1
    local array_index=${2:-${SLURM_ARRAY_TASK_ID}}

    case "${profile}" in
    gmtkn55)
        name_mol_input_list=(
            "molecule_PCONF21" "molecule_S66" "molecule_HEAVY28" "molecule_MCONF" "molecule_BSR36" "molecule_RG18" "molecule_HAL59" "molecule_Amino20x4" "molecule_BH76" "molecule_RSE43" "molecule_S22" "molecule_ISOL24" "molecule_UPU23" "molecule_PNICO23" "molecule_ADIM6" "molecule_BUT14DIOL" "molecule_SIE4x4" "molecule_ACONF" "molecule_IDISP" "molecule_FH51" "molecule_DARC" "molecule_CDIE20" "molecule_TAUT15" "molecule_MB16_43" "molecule_ISO34" "molecule_BHPERI" "molecule_PArel" "molecule_DC13" "molecule_YBDE18" "molecule_ICONF" "molecule_BH76RC" "molecule_SCONF" "molecule_G21EA" "molecule_NBPRC" "molecule_PX13" "molecule_W4_11" "molecule_BHROT27" "molecule_WATER27" "molecule_AL2X6" "molecule_INV24" "molecule_HEAVYSB11" "molecule_G2RC" "molecule_CARBHB12" "molecule_RC21" "molecule_ALK8" "molecule_AHB21" "molecule_WCPT18" "molecule_BHDIV10" "molecule_IL16" "molecule_G21IP" "molecule_ALKBDE10" "molecule_PA26" "molecule_CHB6" "molecule_C60ISO" "molecule_DIPCS10"
        )
        ;;
    intermolecular)
        name_mol_input_list=(
            "molecule_S66x8-090" "molecule_S66x8-095" "molecule_S66x8-100" "molecule_S66x8-105" "molecule_S66x8-110" "molecule_S66x8-125" "molecule_S66x8-150" "molecule_S66x8-200" "molecule_S66x8-A" "molecule_S66x8-B" "molecule_S22x5-0.9" "molecule_S22x5-1.0" "molecule_S22x5-1.2" "molecule_S22x5-1.5" "molecule_S22x5-2.0" "molecule_S22x5-A" "molecule_S22x5-B" "molecule_NCIBLIND10"
        )
        ;;
    nciblind10)
        name_mol_input_list=("molecule_NCIBLIND10")
        ;;
    tmc)
        name_mol_input_list=("TMB28_C1" "TMB28_T1")
        ;;
    *)
        echo "Unknown molecule profile: ${profile}" >&2
        return 2
        ;;
    esac

    export name_mol_input="${name_mol_input_list[${array_index}]}"
}

run_test_job() {
    "${PYTHON_BIN}" "${REPO_ROOT}/test.py" \
        --basis "${basis_args}" \
        --precision float64 \
        ${load_model_args} \
        --name_mol_reverse "${name_mol_reverse}" \
        --dataset "${DATASET}" \
        --name_mol "${name_mol_input}" \
        --if_continue "${if_continue_args}" \
        --if_grad "${IF_GRAD}" \
        "$@"
}

run_model_detail_job() {
    "${PYTHON_BIN}" "${REPO_ROOT}/test_model_detail.py" \
        --basis "${basis_args}" \
        --precision float64 \
        ${load_model_args} \
        --name_mol_reverse "${name_mol_reverse}" \
        --dataset "${DATASET}" \
        --name_mol "${name_mol_input}" \
        --if_continue "${if_continue_args}" \
        "$@"
}

run_cpu_benchmark_job() {
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmark_dft.py" \
        --basis "${basis_args}" \
        --name_mol_reverse "${name_mol_reverse}" \
        --dataset "${DATASET}" \
        --name_mol "${name_mol_input}" \
        --if_continue "${if_continue_args}" \
        --device cpu
}

run_external_benchmark_job() {
    local python_bin=$1
    local script=$2
    shift 2

    "${python_bin}" "${REPO_ROOT}/${script}" "$@"
}
