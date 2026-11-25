#!/usr/bin/env python3
"""Run existing array job scripts locally, respecting array concurrency."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote

if __package__:
    from . import submit_direct_array_per_gpu as submit
else:
    import submit_direct_array_per_gpu as submit


def query_gpus(
    requested: list[str] | None, min_free_memory: float, max_power: float | None,
) -> dict[str, str]:
    def sample() -> dict[str, tuple[str, float, str]]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=15,
        )
        found = set()
        selected = {}
        for line in result.stdout.splitlines():
            index, gpu_uuid, memory, power = (field.strip() for field in line.split(","))
            found.update((index, gpu_uuid))
            if requested is None or index in requested or gpu_uuid in requested:
                selected[index] = (gpu_uuid, float(memory), power)
        if requested is not None and (missing := set(requested) - found):
            raise ValueError(f"Unknown GPU IDs: {', '.join(sorted(missing))}")
        return selected

    snapshot = sample()
    power_sums: dict[str, float] = {}
    if max_power is not None and snapshot:
        start = time.monotonic()
        devices = {index: values[0] for index, values in snapshot.items()}
        for second in range(11):
            if second:
                time.sleep(max(0, start + second - time.monotonic()))
                snapshot = sample()
                if {index: values[0] for index, values in snapshot.items()} != devices:
                    raise RuntimeError("GPU selection changed during power sampling")
            for index, (_, _, power) in snapshot.items():
                try:
                    power_value = float(power)
                except ValueError as error:
                    raise ValueError(f"GPU {index} cannot report power: {power}") from error
                if not math.isfinite(power_value):
                    raise ValueError(f"GPU {index} cannot report power: {power}")
                power_sums[index] = power_sums.get(index, 0.0) + power_value

    available = {}
    for index, (gpu_uuid, memory, _) in snapshot.items():
        if memory < min_free_memory * 1024:
            continue
        if max_power is not None and power_sums[index] / 11 > max_power:
            continue
        # UUIDs avoid CUDA ordinal remapping and CUDA_DEVICE_ORDER assumptions.
        available[index] = gpu_uuid
    return available


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default="test/test_diet100.bash")
    parser.add_argument("--array", help="Override the script's #SBATCH array, e.g. 0-54%%2.")
    parser.add_argument(
        "--gpus", help="Comma-separated nvidia-smi indices or GPU UUIDs; defaults to "
        "CUDA_VISIBLE_DEVICES if set, otherwise all local GPUs. Ignored if "
        "nvidia-smi is missing.",
    )
    parser.add_argument(
        "--array-concurrency", type=int,
        help="Maximum simultaneous jobs per GPU, or total without GPU selection; "
        "overrides the array's %%limit.",
    )
    parser.add_argument("--cpus-per-task", type=int, help="Override #SBATCH CPU count.")
    parser.add_argument("--model-indices", help="Comma-separated MODEL_INDEX values.")
    parser.add_argument(
        "--time-array", help="Optional timing JSON: run longer tasks first; skip zero times.",
    )
    parser.add_argument("--min-free-memory", type=float, default=15.0, help="GiB (default: 15).")
    parser.add_argument(
        "--max-gpu-power", type=float,
        help="Maximum 10-second average power in watts (sampled once per second).",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds (default: 5).")
    parser.add_argument(
        "--wait-for-gpu", action="store_true",
        help="Keep all selected GPUs in the queue, including those currently busy; "
        "wait until each meets the memory and power filters before dispatch.",
    )
    parser.add_argument("--log-dir", help="Default: log/direct-<run timestamp>.")
    parser.add_argument(
        "--foreground", action="store_true",
        help="Stay attached to the terminal instead of starting a background launcher.",
    )
    parser.add_argument("--background-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without starting jobs or creating files. With --gpus, "
        "also skip nvidia-smi and resource checks.",
    )
    return parser


def stop_jobs(running: dict[subprocess.Popen, tuple[str | None, int, int]]) -> None:
    for process in running:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for process in running:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_jobs(
    script: Path, jobs: list[tuple[int, int]], gpus: dict[str, str] | None,
    concurrency: int, cpus: int, log_dir: Path, scratch_dir: Path,
    min_free_memory: float, max_power: float | None, poll_interval: float,
    *, array_task_count: int | None = None,
) -> int:
    pending = deque(jobs)
    resources: list[str | None] = list(gpus) if gpus is not None else [None]
    running: dict[subprocess.Popen, tuple[str | None, int, int]] = {}
    failures = 0
    total_concurrency = concurrency * len(resources)
    waiting_for_gpu = False
    job_status = {
        (model, task): ["-", "-", "Pending", "-", None, None]
        for model, task in jobs
    }
    log_metadata = {
        (model, task): scratch_dir / f"model-{model}-task-{task}" / "log-paths"
        for model, task in jobs
    }
    status_path = log_dir / "launcher-status.md"

    def file_link(path: str | None, label: str) -> str:
        if path is None:
            return "-"
        relative_path = os.path.relpath(path, status_path.parent)
        return f"[{label}]({quote(relative_path, safe='/-._~')})"

    def write_status_table() -> None:
        rows = [
            "| Model | Task | GPU | PID | Status | Exit code | Log | Err |",
            "| ---: | ---: | --- | ---: | --- | ---: | --- | --- |",
        ]
        rows.extend(
            f"| {model} | {task} | {gpu} | {pid} | {status} | {exit_code} "
            f"| {file_link(log_file, 'out')} | {file_link(err_file, 'err')} |"
            for (model, task), (gpu, pid, status, exit_code, log_file, err_file)
            in job_status.items()
        )
        temporary_path = status_path.with_suffix(".md.tmp")
        temporary_path.write_text(
            f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n\n"
            + "\n".join(rows) + "\n"
        )
        temporary_path.replace(status_path)

    write_status_table()
    try:
        while pending or running:
            metadata_updated = False
            for job, metadata_path in log_metadata.items():
                if job_status[job][4] is not None or not metadata_path.exists():
                    continue
                paths = metadata_path.read_text().splitlines()
                if len(paths) == 2:
                    job_status[job][4:] = paths
                    metadata_updated = True
            if metadata_updated:
                write_status_table()
            for process, (gpu, model, task) in list(running.items()):
                if (returncode := process.poll()) is None:
                    continue
                del running[process]
                failures += returncode != 0
                job_status[(model, task)][:4] = [
                    str(gpu) if gpu is not None else "unmanaged",
                    str(process.pid),
                    "Failed" if returncode else "Completed",
                    str(returncode),
                ]
                write_status_table()
                device = f"GPU={gpu}" if gpu is not None else "GPU=unmanaged"
                print(
                    f"[DONE] model={model} task={task} {device} exit={returncode}",
                    flush=True,
                )
            active_by_gpu = {
                gpu: sum(active_gpu == gpu for active_gpu, _, _ in running.values())
                for gpu in resources
            }
            available_gpus = [
                gpu for gpu, active in active_by_gpu.items() if active < concurrency
            ]
            if pending and available_gpus and len(running) < total_concurrency:
                available = (
                    set(query_gpus(list(gpus.values()), min_free_memory, max_power))
                    if gpus is not None else {None}
                )
                if not any(gpu in available for gpu in available_gpus):
                    if not waiting_for_gpu:
                        print(
                            f"[WAIT] {len(pending)} jobs pending; no selected GPU "
                            "currently satisfies the resource filters.",
                            flush=True,
                        )
                        waiting_for_gpu = True
                else:
                    waiting_for_gpu = False
                for gpu in available_gpus:
                    while (
                        pending and active_by_gpu[gpu] < concurrency
                        and len(running) < total_concurrency
                    ):
                        if gpu not in available:
                            break
                        model, task = pending.popleft()
                        stem = f"model-{model}-task-{task}"
                        task_scratch = scratch_dir / stem
                        task_scratch.mkdir(parents=True)
                        env = os.environ.copy()
                        env.update(
                            MODEL_INDEX=str(model), SLURM_ARRAY_TASK_ID=str(task),
                            SLURM_CPUS_PER_TASK=str(cpus),
                            PYSCF_TMPDIR=str(task_scratch),
                            DIRECT_RUN_LOG_LINKS="1",
                            DIRECT_RUN_LOG_METADATA=str(
                                task_scratch / "log-paths"
                            ),
                        )
                        if array_task_count is not None:
                            env["SLURM_ARRAY_TASK_COUNT"] = str(array_task_count)
                        if gpu is not None and gpus is not None:
                            env.update(
                                FORCE_CUDA_VISIBLE_DEVICES=gpus[gpu],
                                CUDA_VISIBLE_DEVICES=gpus[gpu],
                            )
                        process = subprocess.Popen(
                            ["bash", str(script)], cwd=submit.ROOT, env=env,
                            start_new_session=True,
                        )
                        running[process] = (gpu, model, task)
                        active_by_gpu[gpu] += 1
                        job_status[(model, task)][:4] = [
                            str(gpu) if gpu is not None else "unmanaged",
                            str(process.pid),
                            "Running",
                            "-",
                        ]
                        write_status_table()
                        device = f"GPU={gpu}" if gpu is not None else "GPU=unmanaged"
                        print(
                            f"[START] model={model} task={task} {device} PID={process.pid}",
                            flush=True,
                        )
            if pending or running:
                time.sleep(poll_interval)
    finally:
        stop_jobs(running)
    print(f"[SUMMARY] jobs={len(jobs)} failed={failures} logs={log_dir}", flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.background_worker and (not args.foreground or not args.log_dir):
        parser.error("--background-worker requires --foreground and --log-dir")
    for name in ("min_free_memory", "max_gpu_power", "poll_interval"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.poll_interval == 0:
        parser.error("--poll-interval must be positive")
    script = submit.resolve_path(args.script)
    text = script.read_text()
    array = args.array or submit.read_sbatch_field(text, ["-a", "--array"])
    if not array:
        parser.error("Specify --array or include #SBATCH --array in the job script")
    task_ids = submit.parse_array_ids(array)
    if min(task_ids) < 0:
        parser.error("Array task IDs must be non-negative")
    array_task_count = len(task_ids)
    concurrency = args.array_concurrency
    if concurrency is None:
        concurrency = submit.parse_array_limit(array)
    cpus = args.cpus_per_task
    if cpus is None:
        cpus = int(submit.read_sbatch_field(text, ["-c", "--cpus-per-task"]) or 1)
    if cpus < 1 or (concurrency is not None and concurrency < 1):
        parser.error("CPU count and concurrency must be positive")
    models = (
        [int(value) for value in submit.parse_csv_list(args.model_indices)]
        if args.model_indices is not None
        else list(range(len(submit.extract_load_model_args(text)) or 1))
    )
    if not models or min(models) < 0 or len(models) != len(set(models)):
        parser.error("--model-indices must contain distinct non-negative integers")
    if args.time_array:
        with submit.resolve_path(args.time_array).open() as file:
            timings = json.load(file)["time_array"]
        if max(task_ids) >= len(timings):
            parser.error("Array task IDs exceed timing array bounds")
        durations = {task: float(timings[task]) for task in task_ids}
        if any(not math.isfinite(value) or value < 0 for value in durations.values()):
            parser.error("Task times must be finite and non-negative")
        task_ids = sorted(
            (task for task in task_ids if durations[task] > 0),
            key=lambda task: durations[task], reverse=True,
        )
    jobs = [(model, task) for model in models for task in task_ids]
    if not jobs:
        print("[INFO] No jobs remain after filtering zero runtimes.", flush=True)
        return 0
    gpu_text = args.gpus if args.gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    requested = submit.parse_csv_list(gpu_text) if gpu_text is not None else None
    if requested == [] and args.gpus is not None:
        parser.error("The GPU selection is empty")
    if args.dry_run and requested:
        gpus: dict[str, str] | None = dict.fromkeys(requested, "")
    else:
        try:
            gpus = query_gpus(
                requested,
                0 if args.wait_for_gpu else args.min_free_memory,
                None if args.wait_for_gpu else args.max_gpu_power,
            )
        except FileNotFoundError as error:
            if error.filename != "nvidia-smi":
                raise
            if args.wait_for_gpu:
                raise RuntimeError("--wait-for-gpu requires nvidia-smi") from error
            print(
                "[WARN] nvidia-smi not found; GPU selection and filters are "
                "disabled. Jobs inherit CUDA visibility.",
                file=sys.stderr, flush=True,
            )
            gpus = None
        else:
            if requested == []:
                parser.error("The GPU selection is empty")
            if not gpus:
                if args.wait_for_gpu:
                    raise RuntimeError("No selected local GPUs found")
                raise RuntimeError("No local GPUs satisfy the memory/power filters")
    concurrency = concurrency or 1
    total_concurrency = concurrency * (len(gpus) if gpus is not None else 1)
    run_id = datetime.now().strftime("direct-%Y%m%d-%H%M%S-%f")
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else (
        submit.ROOT / "log" / run_id
    )
    scratch_base = Path(
        os.environ.get("PYSCF_TMPDIR") or os.environ.get("TMPDIR") or submit.ROOT / "tmp"
    ).expanduser().resolve()
    scratch_dir = scratch_base / run_id
    selection = (
        f"GPUs={','.join(gpus)} concurrency={concurrency}/GPU"
        if gpus is not None else f"GPU-selection=disabled concurrency={concurrency}"
    )
    print(
        f"[PLAN] script={script} jobs={len(jobs)} {selection} "
        f"total-concurrency={total_concurrency} cpus/job={cpus} "
        f"SLURM_ARRAY_TASK_COUNT={array_task_count} logs={log_dir}",
        flush=True,
    )
    if args.dry_run:
        for model, task in jobs:
            print(f"[DRY-RUN] MODEL_INDEX={model} SLURM_ARRAY_TASK_ID={task}")
        return 0
    if not args.background_worker:
        log_dir.mkdir(parents=True, exist_ok=False)
    if not args.foreground:
        launcher_log = log_dir / "launcher.log"
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()), *argv,
            "--script", str(script),
        ]
        if gpus is not None:
            command.extend(("--gpus", ",".join(gpus.values())))
        command.extend(("--foreground", "--background-worker", "--log-dir", str(log_dir)))
        with launcher_log.open("w") as output:
            output.write(f"[LAUNCHER-LOG] path={launcher_log}\n")
            output.flush()
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            stop_command = f"[STOP] kill -TERM {process.pid}\n"
            output.write(stop_command)
            output.flush()
        (log_dir / "launcher.pid").write_text(f"{process.pid}\n")
        time.sleep(0.2)
        if (
            process.poll() is not None and process.returncode != 0
            and not (log_dir / "exit-code").exists()
        ):
            print(f"[ERROR] Launcher failed; inspect {launcher_log}", flush=True)
            return process.returncode
        print(
            f"[BACKGROUND] PID={process.pid} log={launcher_log}\n"
            f"[MONITOR] tail -f {launcher_log}\n"
            f"[STATUS] {log_dir / 'launcher-status.md'} (auto-updated)\n"
            f"[STOP] kill -TERM {process.pid}", flush=True,
        )
        return 0

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    status = 1
    try:
        status = run_jobs(
            script, jobs, gpus, concurrency, cpus, log_dir, scratch_dir,
            args.min_free_memory, args.max_gpu_power, args.poll_interval,
            array_task_count=array_task_count,
        )
    except KeyboardInterrupt:
        print("[STOP] Interrupted; local job process groups terminated.", flush=True)
        status = 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        (log_dir / "exit-code").write_text(f"{status}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
