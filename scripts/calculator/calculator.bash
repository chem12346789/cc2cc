if (( $# > 1 )) || [[ $# -eq 1 && $1 != --refresh ]]; then
    printf 'Usage: %s [--refresh]\n' "$0" >&2
    exit 2
fi

export basis_args="def2-QZVP-GMTKN55"
# Entries: model:epoch:energy_column:note.
export model_load_epoch_pairs=(
    "atom-82794:10005:scf_d3bj_ene:SCF"
    "atom-3259973:2395:scf_d3bj_ene:SCF"
    "atom-2065162:4955:scf_d3bj_ene:SCF"
)

# export model_load_list=("atom-97269")
# export model_load_list=("atom-1207531")
# export basis_args="def2-QZVPPD"

# export model_load_list=("atom")
# export basis_args="def2-QZVP"

# export model_load_list=("atom-1205822")
# export basis_args="def2-TZVPPD"

find log -type f -mtime +4 \( -name "*.log" -o -name "*.err" -o -name "*.out" \) -exec rm -f {} \;
find log -type d -empty -delete

header_format='| %-30s | %8s | %8s | %9s | %-9s | %-9s | %-13s | %-13s |\n'
table_format='| %-30s | %8s | %8.2f | %9s | %-9s | %-9s | %-13s | %-13s |\n'
printf "$header_format" 'Model' 'Epoch' 'AI' 'Corrected' 'Processed' 'Converged' 'Energy_column' 'Note'
printf '%s\n' '|--------------------------------|----------|----------|-----------|-----------|-----------|---------------|---------------|'
find_messages=()
for model_load_epoch_pair in "${model_load_epoch_pairs[@]}"; do
    IFS=: read -r model_load load_epoch energy_column note <<< "$model_load_epoch_pair"
    energy_column="${energy_column:-scf_d3bj_ene}"
    note="${note:-}"
    log_file="log/collect_info_${model_load}_${load_epoch}_${energy_column}_gmtkn55_cal_full.log"

    rm -rf validate/*.csv
    find_output=$(find "validate/${basis_args}_${model_load}/${load_epoch}" -maxdepth 1 -type f -exec cp -t validate -- {} + 2>&1)
    if [[ -n "$find_output" ]]; then
        find_messages+=("$find_output")
    fi

    ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load "$model_load" --epoch "$load_epoch" --energy_column "$energy_column" --basis "$basis_args" --verbose 4 --frequency 30m --max_checks 0 --data_set gmtkn-def2 "$@" >"$log_file" 2>&1
    awk -v model="${model_load#atom-}" -v epoch="$load_epoch" -v energy_column="$energy_column" -v note="$note" -v table_format="$table_format" -v log_file="$log_file" '
        BEGIN {
            corrected_header = energy_column
            if (energy_column == "scf_d3bj_ene") corrected_header = "AI(BJ)"
            else if (energy_column == "scf_ene") corrected_header = "AI"
            else if (energy_column == "modified_ai_d3bj") corrected_header = "AI(M"
        }
        /^Saving excel backup files with timestamp:/ { finished = 1 }
        !finished && $NF == "Processed" {
            ai_field = corrected_field = truncated = 0
            for (i = 1; i <= NF; i++) {
                if ($i == "AI") ai_field = i + 1
                if ($i == corrected_header) corrected_field = i + 1
                if ($i == "...") truncated = 1
            }
        }
        !finished && $1 == "summary" && ai_field && corrected_field && !truncated {
            ai = $(ai_field)
            corrected = $(corrected_field)
            processed = $NF
        }
        /^Converged tests: / { converged = $3 }
        END {
            if (ai != "" && corrected != "") {
                energy_label = energy_column
                if (energy_column == "scf_d3bj_ene") {
                    energy_label = "d3bj"
                } else if (energy_column == "post_scf_vv10_ene") {
                    energy_label = "post_vv10"
                }
                printf table_format, model, epoch, ai, corrected, processed, converged, energy_label, note
            } else {
                printf "No complete summary found for %s epoch %s; see %s\n", model, epoch, log_file > "/dev/stderr"
                exit 1
            }
        }
    ' "$log_file"
done
if ((${#find_messages[@]})); then
    printf '%s\n' "${find_messages[@]}" >&2
fi
rm -rf validate/*.csv
