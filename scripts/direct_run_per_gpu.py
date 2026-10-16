#!/usr/bin/env python3
"""Run existing array job scripts locally, with one active job per GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

if __package__:
    from . import submit_direct_array_per_gpu as submit
else:
    import submit_direct_array_per_gpu as submit


def query_gpus(
    requested: list[str] | None, min_free_memory: float, max_power: float | None,
) -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,power.draw",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=15,
    )
    available = {}
    found = set()
    for line in result.stdout.splitlines():
        index, gpu_uuid, memory, power = (field.strip() for field in line.split(","))
        found.update((index, gpu_uuid))
        if requested is not None and index not in requested and gpu_uuid not in requested:
            continue
        if float(memory) < min_free_memory * 1024:
            continue
        if max_power is not None:
            try:
                power_value = float(power)
            except ValueError as error:
                raise ValueError(f"GPU {index} cannot report power: {power}") from error
            if not math.isfinite(power_value) or power_value > max_power:
                continue
        # UUIDs avoid CUDA ordinal remapping and CUDA_DEVICE_ORDER assumptions.
        available[index] = gpu_uuid
    if requested is not None and (missing := set(requested) - found):
        raise ValueError(f"Unknown GPU IDs: {', '.join(sorted(missing))}")
    return available


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default="test/test_diet100.bash")
    parser.add_argument("--array", help="Override the script's #SBATCH array, e.g. 0-54%%2.")
    parser.add_argument(
        "--gpus", help="Comma-separated nvidia-smi indices or GPU UUIDs; defaults to "
        "CUDA_VISIBLE_DEVICES if set, otherwise all local GPUs.",
    )
    parser.add_argument(
        "--array-concurrency", type=int,
        help="Maximum simultaneous jobs; overrides the array's %%limit.",
    )
    parser.add_argument("--cpus-per-task", type=int, help="Override #SBATCH CPU count.")
    parser.add_argument("--model-indices", help="Comma-separated MODEL_INDEX values.")
    parser.add_argument(
        "--time-array", help="Optional timing JSON: run longer tasks first; skip zero times.",
    )
    parser.add_argument("--min-free-memory", type=float, default=15.0, help="GiB (default: 15).")
    parser.add_argument("--max-gpu-power", type=float, help="Maximum power in watts.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds (default: 5).")
    parser.add_argument("--log-dir", help="Default: log/direct-<unique run ID>.")
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


def stop_jobs(running: dict[subprocess.Popen, tuple[str, int, int]]) -> None:
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
    script: Path, jobs: list[tuple[int, int]], gpus: dict[str, str],
    concurrency: int, cpus: int, log_dir: Path, scratch_dir: Path,
    min_free_memory: float, max_power: float | None, poll_interval: float,
) -> int:
    pending = deque(jobs)
    running: dict[subprocess.Popen, tuple[str, int, int]] = {}
    failures = 0
    try:
        while pending or running:
            for process, (gpu, model, task) in list(running.items()):
                if (returncode := process.poll()) is None:
                    continue
                del running[process]
                failures += returncode != 0
                print(
                    f"[DONE] model={model} task={task} GPU={gpu} exit={returncode}",
                    flush=True,
                )
            busy = {gpu for gpu, _, _ in running.values()}
            free = [gpu for gpu in gpus if gpu not in busy]
            if pending and free and len(running) < concurrency:
                available = query_gpus(list(gpus.values()), min_free_memory, max_power)
                for gpu in free:
                    if not pending or len(running) >= concurrency:
                        break
                    if gpu not in available:
                        continue
                    model, task = pending.popleft()
                    stem = f"model-{model}-task-{task}"
                    task_scratch = scratch_dir / stem
                    task_scratch.mkdir(parents=True)
                    env = os.environ.copy()
                    env.update(
                        MODEL_INDEX=str(model), SLURM_ARRAY_TASK_ID=str(task),
                        SLURM_CPUS_PER_TASK=str(cpus),
                        FORCE_CUDA_VISIBLE_DEVICES=gpus[gpu],
                        CUDA_VISIBLE_DEVICES=gpus[gpu],
                        PYSCF_TMPDIR=str(task_scratch),
                    )
                    with (log_dir / f"{stem}.out").open("w") as stdout, (
                        log_dir / f"{stem}.err"
                    ).open("w") as stderr:
                        process = subprocess.Popen(
                            ["bash", str(script)], cwd=submit.ROOT, env=env,
                            stdout=stdout, stderr=stderr, start_new_session=True,
                        )
                    running[process] = (gpu, model, task)
                    print(
                        f"[START] model={model} task={task} GPU={gpu} PID={process.pid}",
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
    if requested == []:
        parser.error("The GPU selection is empty")
    gpus = (
        dict.fromkeys(requested, "") if args.dry_run and requested is not None
        else query_gpus(requested, args.min_free_memory, args.max_gpu_power)
    )
    if not gpus:
        raise RuntimeError("No local GPUs satisfy the memory/power filters")
    concurrency = min(concurrency or len(gpus), len(gpus))
    run_id = f"direct-{uuid.uuid4().hex[:12]}"
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else (
        submit.ROOT / "log" / run_id
    )
    scratch_base = Path(
        os.environ.get("PYSCF_TMPDIR") or os.environ.get("TMPDIR") or submit.ROOT / "tmp"
    ).expanduser().resolve()
    scratch_dir = scratch_base / run_id
    print(
        f"[PLAN] script={script} jobs={len(jobs)} GPUs={','.join(gpus)} "
        f"concurrency={concurrency} cpus/job={cpus} logs={log_dir}", flush=True,
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
            "--script", str(script), "--gpus", ",".join(gpus.values()),
            "--foreground", "--background-worker", "--log-dir", str(log_dir),
        ]
        with launcher_log.open("w") as output:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
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
