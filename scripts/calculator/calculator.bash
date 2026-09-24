export basis_args="def2-QZVP(D)"
export model_load_epoch_pairs=(
    # "atom-dft:1000"
    # "atom-82794:475"
    # "atom-82794:1115"
    # "atom-82794:2395"
    # "atom-82794:4940"
    "atom-82794:10005"
    # "atom-1786123:1115"
    # "atom-1786123:2390"
    # "atom-2647768:625"
)

# export model_load_list=("atom-97269")
# export model_load_list=("atom-1207531")
# export basis_args="def2-QZVPPD"

# export model_load_list=("atom")
# export basis_args="def2-QZVP"

# export model_load_list=("atom-1205822")
# export basis_args="def2-TZVPPD"

for model_load_epoch_pair in "${model_load_epoch_pairs[@]}"; do
    model_load="${model_load_epoch_pair%%:*}"
    load_epoch="${model_load_epoch_pair##*:}"
    load_csv="/home/chenzihao/workspace/cc2cc_test5/validate_hkqai_done/ccdft_def2-QZVP(D)_atom-82794_gmtkn-def2.csv"

    rm -rf validate/*.csv
    find "validate/${basis_args}_${model_load}/${load_epoch}" -maxdepth 1 -type f -exec cp -t validate -- {} +

    echo "Starting data collection for model ${model_load} with epoch ${load_epoch} and basis ${basis_args}..."
    ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load "${model_load}" --epoch "${load_epoch}" --basis "${basis_args}" --verbose 4 --data_set gmtkn-def2 --frequency 30m --max_checks 0 --load_csv "${load_csv}" >"log/collect_info_${model_load}_${load_epoch}.log" 2>&1
    cat log/collect_info_${model_load}_${load_epoch}.log | grep "summary" | tail -n 1
done
