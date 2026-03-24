import os
import time

import wandb
import argparse


if __name__ == "__main__":
    wandb.init(project="monitor_pid_men_cpu", name="monitor_pid_men_cpu")
    print("Initialized Weights & Biases project: monitor_pid_men_cpu")
    print(f"you can kill it by running: kill {os.getpid()}", flush=True)

    # Read PIDs from the input argument
    parser = argparse.ArgumentParser(
        description="Monitor memory and CPU usage of specified PIDs."
    )
    parser.add_argument(
        "--pids",
        nargs="+",
        type=int,
        required=True,
        help="List of PIDs to monitor.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="10s",
        help="Interval between checks (e.g., 10s, 1m, 1h). Default is 10 seconds.",
    )

    args = parser.parse_args()
    # Convert interval to seconds
    if args.interval.endswith("s"):
        args.interval = int(args.interval[:-1])
    elif args.interval.endswith("m"):
        args.interval = int(args.interval[:-1]) * 60
    elif args.interval.endswith("h"):
        args.interval = int(args.interval[:-1]) * 3600
    else:
        args.interval = int(args.interval)

    while True:
        wandb_log = {}
        for pid in args.pids:

            # Extract CPU usage from the top command output
            result = os.popen(f"top -b -n 1 -p {pid}").readlines()
            if len(result) < 8:
                args.pids.remove(pid)
                continue
            else:
                cpu_usage = result[7].split()[8]  # CPU usage in %
                wandb_log[f"PID_{pid}_CPU_Usage"] = float(cpu_usage)

            # Extract Memory usage from the proc filesystem
            if os.path.exists(f"/proc/{pid}/status"):
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            mem_usage_kb = int(line.split()[1])  # Memory usage in KB
                            wandb_log[f"PID_{pid}_Memory_Usage_GB"] = (
                                mem_usage_kb / 1024 / 1024
                            )  # Convert to GB
                            break

        if not args.pids:
            print("All monitored PIDs have finished.")
            break

        wandb.log(wandb_log)
        time.sleep(args.interval)
