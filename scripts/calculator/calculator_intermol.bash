if [[ ${1:-} == --check-log ]]; then
    if (( $# != 2 )); then
        printf 'Usage: %s --check-log LOG_FILE\n' "$0" >&2
        exit 2
    fi
    if [[ ! -f $2 || ! -r $2 ]]; then
        printf 'Cannot read log file: %s\n' "$2" >&2
        exit 1
    fi
    awk -v log_file="$2" '
        /^model_load: / { model = substr($0, 13) }
        /^epoch: / { epoch = substr($0, 8) }
        /^basis: / { basis = substr($0, 8) }
        /^energy_column: / { energy_column = substr($0, 16) }
        /^Total number of reactions process\/done: / {
            processed = $NF
            summary_line = NR
        }
        /^Converged tests: / { converged = $3; converged_line = NR }
        /^All data processed\. Exiting\.$/ { finished_line = NR }
        END {
            if (processed != "DONE" || !summary_line ||
                converged_line <= summary_line || finished_line <= converged_line) {
                printf "Incomplete intermolecular log: %s\n", log_file > "/dev/stderr"
                exit 1
            }
            if (converged !~ /^[0-9]+\/[0-9]+$/) {
                printf "Convergence counts unavailable in %s\n", log_file > "/dev/stderr"
                exit 1
            }
            split(converged, counts, "/")
            if (counts[2] == 0 || counts[1] != counts[2]) {
                printf "Not all tests converged (%s) in %s\n", converged, log_file > "/dev/stderr"
                exit 1
            }
            printf "Model: %s\nEpoch: %s\nBasis: %s\nEnergy column: %s\nProcessed: %s\nConverged: %s\n",
                model, epoch, basis, energy_column, processed, converged
        }
    ' "$2"
    exit $?
fi

if (( $# > 1 )) || [[ $# -eq 1 && $1 != --refresh ]]; then
    printf 'Usage: %s [--refresh] | --check-log LOG_FILE\n' "$0" >&2
    exit 2
fi

export basis_args="def2-QZVP(D)"
# export basis_args="def2-QZVPPD"
export VALIDATE_DIR="validate_hkqai_done"
# Entries: model:epoch:energy_column:note.
export model_load_epoch_pairs=(
    "atom-3259973:2395:scf_d3bj_ene:"
)

find log -type f -mtime +4 \( -name "*.log" -o -name "*.err" -o -name "*.out" \) -exec rm -f {} \;
find log -type d -empty -delete

header_format='| %-30s | %8s | %-9s | %-9s | %-13s | %-13s |\n'
table_format='| %-30s | %8s | %-9s | %-9s | %-13s | %-13s |\n'
printf "$header_format" 'Model' 'Epoch' 'Processed' 'Converged' 'Energy_column' 'Note'
printf '%s\n' '|--------------------------------|----------|-----------|-----------|---------------|---------------|'
find_messages=()
for model_load_epoch_pair in "${model_load_epoch_pairs[@]}"; do
    IFS=: read -r model_load load_epoch energy_column note <<< "$model_load_epoch_pair"
    energy_column="${energy_column:-scf_d3bj_ene}"
    note="${note:-}"
    csv_energy_suffix=""
    if [[ "$energy_column" != "scf_d3bj_ene" ]]; then
        csv_energy_suffix="_${energy_column}"
    fi
    log_file="log/intermol_${model_load}_${load_epoch}_${energy_column}.log"

    rm -rf validate/*.csv
    find_output=$(find "validate/${basis_args}_${model_load}/${load_epoch}" -maxdepth 1 -type f -exec cp -t validate -- {} + 2>&1)
    if [[ -n "$find_output" ]]; then
        find_messages+=("$find_output")
    fi

    if ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load "$model_load" --epoch "$load_epoch" --energy_column "$energy_column" --basis "$basis_args" --verbose 0 --frequency 30m --max_checks 0 --data_set dft-fitset-def2 "$@" >"$log_file" 2>&1; then
        :
    else
        status=$?
        printf 'Data collection failed for %s epoch %s; see %s\n' "$model_load" "$load_epoch" "$log_file" >&2
        exit "$status"
    fi

    if ! awk -v model="${model_load#atom-}" -v epoch="$load_epoch" -v energy_column="$energy_column" -v note="$note" -v table_format="$table_format" -v log_file="$log_file" '
        /^Total number of reactions process\/done: / { processed = $NF }
        /^Converged tests: / { converged = $3 }
        END {
            if (processed == "DONE") {
                printf table_format, model, epoch, processed, converged, energy_column, note
            } else {
                printf "Intermolecular data is incomplete (%s) for %s epoch %s; see %s\n", processed, model, epoch, log_file > "/dev/stderr"
                exit 1
            }
        }
    ' "$log_file"; then
        exit 1
    fi

    if ! cp -- \
        "$VALIDATE_DIR/ccdft_${basis_args}_${model_load}_${load_epoch}_dft-fitset-def2${csv_energy_suffix}.csv" \
        "$VALIDATE_DIR/ccdft_${basis_args}_${model_load}_dft-fitset-def2.csv"; then
        printf 'Could not stage intermol data for D3 fitting; check VALIDATE_DIR=%s\n' "$VALIDATE_DIR" >&2
        exit 1
    fi
done
if ((${#find_messages[@]})); then
    printf '%s\n' "${find_messages[@]}" >&2
fi
rm -rf validate/*.csv
