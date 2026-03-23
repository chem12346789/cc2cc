import os
import time

import wandb
import argparse


if __name__ == "__main__":
    wandb.init(project="monitor_pid_men_cpu", name="monitor_pid_men_cpu")

    # Read PIDs from the input argument
    parser = argparse.ArgumentParser(
        description="Monitor memory and CPU usage of specified PIDs."
    )
    parser.add_argument(
        "--pids", nargs="+", type=int, required=True, help="List of PIDs to monitor."
    )
    args = parser.parse_args()

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
                            wandb_log[f"PID_{pid}_Memory_Usage_MB"] = (
                                mem_usage_kb / 1024
                            )  # Convert to MB
                            break

        if not args.pids:
            print("All monitored PIDs have finished.")
            break

        wandb.log(wandb_log)
        time.sleep(10)
