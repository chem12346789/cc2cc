export pid="1965833 1965834 2208956 2168612"

echo "Starting monitor for PIDs: $pid"
nohup bash -c "
    echo \"Running monitor at \$(date)...\"
    ~/anaconda3/envs/pyscf/bin/python monitor_pid_men_cpu.py --pids ${pid} --interval 2m
" >log/collect_info.log 2>log/collect_info.err &
echo "Monitor started in the background. Check log/collect_info.log for progress."
