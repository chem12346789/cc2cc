# CLUSTER-ONLY: This script is configured for this cluster; do not run it elsewhere.

if [[ "$(hostname)" == "user-MZ73-LM1-000" ]]; then
    :
else
    printf 'Error: this script can only run on user-MZ73-LM1-000 (current host: %s).\n' "$(hostname)" >&2
    exit 1
fi

export pid="1965833 1965834 2208956 2168612"

echo "Starting monitor for PIDs: $pid"
nohup bash -c "
    echo \"Running monitor at \$(date)...\"
    ~/anaconda3/envs/pyscf/bin/python monitor_pid_men_cpu.py --pids ${pid} --interval 2m
" >log/collect_info.log 2>log/collect_info.err &
echo "Monitor started in the background. Check log/collect_info.log for progress."
