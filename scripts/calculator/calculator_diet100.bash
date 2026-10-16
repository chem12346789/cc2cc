export basis_args="def2-QZVP(D)"
export model_load_epoch_pairs=(
    "atom-dft:1000"
    "atom-82794:1115"
    "atom-82794:2395"
    "atom-82794:10005"
    "atom-1221273:4955"
    "atom-1407441:1115"
    "atom-1407441:2395"
    "atom-7297:1115"
)

# find log folder with 4 days old files
find log -type f -mtime +4 -name "*.log" -exec rm -f {} \;
find log -type f -mtime +4 -name "*.err" -exec rm -f {} \;
find log -type f -mtime +4 -name "*.out" -exec rm -f {} \;
# clean up empty folders in log folder
find log -type d -empty -delete

table_format='%-18s %8s %14s %14s %12s %12s  %s\n'
printf "$table_format" 'Model' 'Epoch' 'AI' 'AI(BJ)' 'Processed' 'Converged' 'Basis'
for pair in "${model_load_epoch_pairs[@]}"; do
    model_load="${pair%%:*}"
    load_epoch="${pair#*:}"

    rm -rf validate/*.csv
    find "validate/${basis_args}_${model_load}/${load_epoch}" -maxdepth 1 -type f -exec cp -t validate -- {} +

    ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load ${model_load} --epoch ${load_epoch} --basis ${basis_args} --verbose 0 --frequency 30m --max_checks 0 --data_set gmtkn-diet100-def2 >log/collect_info_${model_load}_${load_epoch}_cal_100.log 2>&1
    awk -v model="$model_load" -v epoch="$load_epoch" -v basis="$basis_args" -v table_format="$table_format" '
        /summary/ { summary = $0 }
        /^Converged tests: / { converged = $3 }
        END {
            if (summary != "") {
                $0 = summary
                printf table_format, model, epoch, $2, $3, $4, converged, basis
            }
        }
    ' "log/collect_info_${model_load}_${load_epoch}_cal_100.log"
done
rm -rf validate/*.csv
