export basis_args="def2-QZVP(D)"
# Entries: model:epoch:energy_column (e.g. atom-dft:1000:post_scf_vv10_ene).
export model_load_epoch_pairs=(
    "atom-dft:1000:scf_d3bj_ene:B3LYP"
    "atom-82794:1115:scf_d3bj_ene:best before"
    "atom-82794:2395:scf_d3bj_ene:best before"
    "atom-82794:4940:scf_d3bj_ene:best before"
    "atom-82794:10005:scf_d3bj_ene:best before"
    # "atom-7297:1115:scf_d3bj_ene:L1 mol1"
    # "atom-7297:2395:scf_d3bj_ene:L1 mol1"
    # "atom-7297:4955:scf_d3bj_ene:L1 mol1"
    # "atom-4113237:1115:scf_d3bj_ene:L1 mol1"
    # "atom-4113237:2395:scf_d3bj_ene:L1 mol1"
    # "atom-4113237:4955:scf_d3bj_ene:L1 mol1"
    # "atom-3259973:1115:post_scf_vv10_ene:vv10"
    # "atom-3259973:1115:scf_d3bj_ene:vv10"
    "atom-3259973:2395:post_scf_vv10_ene:vv10"
    "atom-3259973:2395:scf_d3bj_ene:vv10"
    "atom-4114139:2395:scf_d3bj_ene:L1 mol2"
)

# find log folder with 4 days old files
find log -type f -mtime +4 -name "*.log" -exec rm -f {} \;
find log -type f -mtime +4 -name "*.err" -exec rm -f {} \;
find log -type f -mtime +4 -name "*.out" -exec rm -f {} \;
# clean up empty folders in log folder
find log -type d -empty -delete

header_format='| %-18s | %8s | %8s | %9s | %-9s | %-9s | %-13s | %-13s |\n'
table_format='| %-18s | %8s | %8.2f | %9.2f | %-9s | %-9s | %-13s | %-13s |\n'
printf "$header_format" 'Model' 'Epoch' 'AI' 'Corrected' 'Processed' 'Converged' 'Energy_column' 'Note'
printf '%s\n' '|--------------------|----------|----------|-----------|-----------|-----------|---------------|---------------|'
find_messages=()
for pair in "${model_load_epoch_pairs[@]}"; do
    IFS=: read -r model_load load_epoch energy_column note <<< "$pair"
    log_file="log/collect_info_${model_load}_${load_epoch}_${energy_column}_cal_100.log"

    rm -rf validate/*.csv
    find_output=$(find "validate/${basis_args}_${model_load}/${load_epoch}" -maxdepth 1 -type f -exec cp -t validate -- {} + 2>&1)
    if [[ -n "$find_output" ]]; then
        find_messages+=("$find_output")
    fi

    ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load "$model_load" --epoch "$load_epoch" --energy_column "$energy_column" --basis "$basis_args" --verbose 0 --frequency 30m --max_checks 0 --data_set gmtkn-diet100-def2 >"$log_file" 2>&1
    awk -v model="$model_load" -v epoch="$load_epoch" -v energy_column="$energy_column" -v note="$note" -v table_format="$table_format" '
        /summary/ { summary = $0 }
        /^Converged tests: / { converged = $3 }
        END {
            if (summary != "") {
                $0 = summary
                energy_label = energy_column
                if (energy_column == "scf_d3bj_ene") {
                    energy_label = "d3bj"
                } else if (energy_column == "post_scf_vv10_ene") {
                    energy_label = "post_vv10"
                }
                printf table_format, model, epoch, $2, $3, $4, converged, energy_label, note
            }
        }
    ' "$log_file"
done
if ((${#find_messages[@]})); then
    printf '%s\n' "${find_messages[@]}" >&2
fi
rm -rf validate/*.csv
