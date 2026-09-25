export basis_args="def2-QZVP(D)"
export model_load_epoch_pairs=(
    "atom-dft:1000"
    "atom-82794:1115"
    "atom-82794:2395"
    "atom-82794:4940"
    "atom-82794:10005"
    "atom-1560444:4955"
    "atom-1563701:4955"
    "atom-1824146:4940"
    "atom-1822511:4955"
)

# find log folder with 4 days old files
find log -type f -mtime +4 -name "*.log" -exec rm -f {} \;
find log -type f -mtime +4 -name "*.err" -exec rm -f {} \;
find log -type f -mtime +4 -name "*.out" -exec rm -f {} \;
# clean up empty folders in log folder
find log -type d -empty -delete

for pair in "${model_load_epoch_pairs[@]}"; do
    model_load="${pair%%:*}"
    load_epoch="${pair#*:}"

    rm -rf validate/*.csv
    find "validate/${basis_args}_${model_load}/${load_epoch}" -maxdepth 1 -type f -exec cp -t validate -- {} +

    echo "Starting data collection for model ${model_load} with epoch ${load_epoch} and basis ${basis_args}..."
    ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load ${model_load} --epoch ${load_epoch} --basis ${basis_args} --verbose 0 --frequency 30m --max_checks 0 --data_set gmtkn-diet100-def2 >log/collect_info_${model_load}_${load_epoch}_cal_100.log 2>&1
    cat log/collect_info_${model_load}_${load_epoch}_cal_100.log | grep "summary" | tail -n 1
done
rm -rf validate/*.csv
