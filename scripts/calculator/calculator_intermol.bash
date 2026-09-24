export basis_args="def2-QZVP(D)"
# export basis_args="def2-QZVPPD"
export model_load_epoch_pairs=(
    "atom-2647768:1265"
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
    ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load ${model_load} --epoch ${load_epoch} --basis ${basis_args} --verbose 0 --frequency 30m --data_set dft-fitset-def2 --max_checks 0 >log/intermol_${model_load}_${load_epoch}.log 2>&1
    cat log/intermol_${model_load}_${load_epoch}.log | grep "summary" | tail -n 1
done
rm -rf validate/*.csv
