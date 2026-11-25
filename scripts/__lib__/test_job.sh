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

set_gpu_numa_node() {
    local -A gpu_numa=()
    local gpu="${CUDA_VISIBLE_DEVICES:-}"
    local gpu_index=""
    local node=""

    if [[ -z "${GPU_NUMA_NODE_MAP:-}" ]]; then
        unset NUMA_NODE
        echo "NUMA binding disabled by GPU_NUMA_NODE_MAP"
        return 0
    fi

    local mapping
    local -a mappings
    IFS=, read -r -a mappings <<< "${GPU_NUMA_NODE_MAP}"
    for mapping in "${mappings[@]}"; do
        if [[ ! "${mapping}" =~ ^([0-9]+):([0-9]+)$ ]]; then
            echo "Invalid GPU_NUMA_NODE_MAP entry: ${mapping}" >&2
            return 1
        fi
        gpu_numa["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
    done

    if [[ "${gpu}" =~ ^[0-9]+$ ]]; then
        gpu_index="${gpu}"
    elif [[ -n "${gpu}" && "${gpu}" != *,* ]]; then
        local gpu_info
        local index
        local uuid
        gpu_info="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)" || {
            echo "Unable to resolve GPU UUID ${gpu} to a device index" >&2
            return 1
        }
        while IFS=, read -r index uuid; do
            index="${index//[[:space:]]/}"
            uuid="${uuid//[[:space:]]/}"
            if [[ "${uuid}" == "${gpu}" ]]; then
                gpu_index="${index}"
                break
            fi
        done <<< "${gpu_info}"
    fi

    if [[ -n "${gpu_index}" ]]; then
        node="${gpu_numa[${gpu_index}]:-}"
    fi
    if [[ -z "${node}" ]]; then
        echo "No NUMA node mapping for CUDA_VISIBLE_DEVICES=${gpu:-unset}" >&2
        return 1
    fi

    if ! command -v numactl >/dev/null 2>&1 ||
        ! numactl "--cpunodebind=${node}" "--preferred=${node}" true >/dev/null 2>&1; then
        echo "Warning: NUMA node ${node} is unavailable; continuing without NUMA binding." >&2
        unset NUMA_NODE
        return 0
    fi

    export NUMA_NODE="${node}"
    echo "NUMA node for GPU ${gpu} (index ${gpu_index}): ${NUMA_NODE}"
}

setup_gpu_test_job() {
    local max_memory_mb=$1
    local min_free_memory_mb=$2
    local use_numba=${3:-0}

    export MAX_MEMORY_GPU="${min_free_memory_mb}"
    setup_test_job "${max_memory_mb}" "${use_numba}"
    select_gpu "${min_free_memory_mb}"
    set_gpu_numa_node || exit 1
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
    (gmtkn55)
        name_mol_input_list=(
            "molecule_PCONF21" "molecule_S66" "molecule_HEAVY28" "molecule_MCONF" "molecule_BSR36" "molecule_RG18" "molecule_HAL59" "molecule_Amino20x4" "molecule_BH76" "molecule_RSE43" "molecule_S22" "molecule_ISOL24" "molecule_UPU23" "molecule_PNICO23" "molecule_ADIM6" "molecule_BUT14DIOL" "molecule_SIE4x4" "molecule_ACONF" "molecule_IDISP" "molecule_FH51" "molecule_DARC" "molecule_CDIE20" "molecule_TAUT15" "molecule_MB16_43" "molecule_ISO34" "molecule_BHPERI" "molecule_PArel" "molecule_DC13" "molecule_YBDE18" "molecule_ICONF" "molecule_BH76RC" "molecule_SCONF" "molecule_G21EA" "molecule_NBPRC" "molecule_PX13" "molecule_W4_11" "molecule_BHROT27" "molecule_WATER27" "molecule_AL2X6" "molecule_INV24" "molecule_HEAVYSB11" "molecule_G2RC" "molecule_CARBHB12" "molecule_RC21" "molecule_ALK8" "molecule_AHB21" "molecule_WCPT18" "molecule_BHDIV10" "molecule_IL16" "molecule_G21IP" "molecule_ALKBDE10" "molecule_PA26" "molecule_CHB6" "molecule_C60ISO" "molecule_DIPCS10"
        )
        ;;
    (intermolecular)
        name_mol_input_list=(
            "molecule_S66x8-090" "molecule_S66x8-095" "molecule_S66x8-100" "molecule_S66x8-105" "molecule_S66x8-110" "molecule_S66x8-125" "molecule_S66x8-150" "molecule_S66x8-200" "molecule_S66x8-A" "molecule_S66x8-B" "molecule_S22x5-0.9" "molecule_S22x5-1.0" "molecule_S22x5-1.2" "molecule_S22x5-1.5" "molecule_S22x5-2.0" "molecule_S22x5-A" "molecule_S22x5-B" "molecule_NCIBLIND10"
        )
        ;;
    (nciblind10)
        name_mol_input_list=("molecule_NCIBLIND10")
        ;;
    (force)
        name_mol_input_list=("molecule_Force")
        ;;
    (rotate)
        name_mol_input_list=("molecule_Rotation")
        ;;
    (tmc)
        name_mol_input_list=("TMB28_C1" "TMB28_T1")
        ;;
    (detail)
        name_mol_input_list=("W4_11-ch4" "W4_11-clf" "W4_11-hcl" "W4_11-h2o" "W4_11-nh3" "W4_11-oh" "W4_11-f2" "W4_11-co" "W4_11-n2" "W4_11-o2" "W4_11-c2h2" "W4_11-c2h4" "W4_11-c2h6" "W4_11-c2")
        ;;
    (*)
        echo "Unknown molecule profile: ${profile}" >&2
        return 2
        ;;
    esac

    export name_mol_input="${name_mol_input_list[${array_index}]}"

    local load_name="no-load"
    local load_epoch="no-epoch"
    local test_mode="${TEST_MODE:-scf}"
    if [[ "${load_model_args:-}" =~ --load[[:space:]=]+([^[:space:]]+) ]]; then
        load_name="${BASH_REMATCH[1]}"
    fi
    if [[ "${load_model_args:-}" =~ --load_epoch[[:space:]=]+([^[:space:]]+) ]]; then
        load_epoch="${BASH_REMATCH[1]}"
    fi
    if [[ "${load_model_args:-}" =~ --test_mode[[:space:]=]+([^[:space:]]+) ]]; then
        test_mode="${BASH_REMATCH[1]}"
    fi
    if [[ "${test_mode}" != "scf" ]]; then
        load_name="${load_name}-${test_mode}"
    fi
    local set_name="${name_mol_input#molecule_}"
    local log_dir="${REPO_ROOT}/log/${load_name}/${load_epoch}"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/${set_name}_${array_index}.out"
    local err_file="${log_dir}/${set_name}_${array_index}.err"
    if [[ "${DIRECT_RUN_LOG_LINKS:-}" == "1" ]]; then
        printf '[TASK-LOG] model=%s task=%s log=%s err=%s\n' \
            "${MODEL_INDEX:-0}" "${array_index}" "${log_file}" "${err_file}"
    fi
    exec >"${log_file}" 2>"${err_file}"
    printf 'array_index=%s\n' "${array_index}"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        printf 'scancel %s_%s\n' "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        printf 'scancel %s\n' "${SLURM_JOB_ID}"
    fi
}

run_test_job() {
    local -a numa_prefix=()
    if [[ -n "${NUMA_NODE:-}" ]]; then
        numa_prefix=(
            numactl
            "--cpunodebind=${NUMA_NODE}"
            "--preferred=${NUMA_NODE}"
        )
    fi

    OMP_WAIT_POLICY=PASSIVE "${numa_prefix[@]}" "${PYTHON_BIN}" \
        "${REPO_ROOT}/test.py" \
        --test_mode "${TEST_MODE:-scf}" \
        --basis "${basis_args}" \
        --precision float64 \
        ${load_model_args} \
        --name_mol_reverse "${name_mol_reverse}" \
        --dataset "${DATASET}" \
        --name_mol "${name_mol_input}" \
        --if_continue "${if_continue_args}" \
        --if_grad "${IF_GRAD}" \
        --grid_response "${GRID_RESPONSE:-0}" \
        --max_memory_gpu "${MAX_MEMORY_GPU:-4000}" \
        "$@"
}

run_cpu_benchmark_job() {
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmark.py" \
        --basis "${basis_args}" \
        --name_mol_reverse "${name_mol_reverse}" \
        --dataset "${DATASET}" \
        --name_mol "${name_mol_input}" \
        --if_continue "${if_continue_args}" \
        --device cpu \
        "$@"
}

run_gpu_benchmark_job() {
    local -a numa_prefix=()
    if [[ -n "${NUMA_NODE:-}" ]]; then
        numa_prefix=(
            numactl
            "--cpunodebind=${NUMA_NODE}"
            "--preferred=${NUMA_NODE}"
        )
    fi

    OMP_WAIT_POLICY=PASSIVE "${numa_prefix[@]}" "${PYTHON_BIN}" \
        "${REPO_ROOT}/benchmark.py" \
        --basis "${basis_args}" \
        --name_mol_reverse "${name_mol_reverse}" \
        --dataset "${DATASET}" \
        --name_mol "${name_mol_input}" \
        --if_continue "${if_continue_args}" \
        --if_grad "${IF_GRAD}" \
        --device cuda \
        "$@"
}

run_external_benchmark_job() {
    local python_bin=$1
    local script=$2
    shift 2

    "${python_bin}" "${REPO_ROOT}/${script}" "$@"
}
