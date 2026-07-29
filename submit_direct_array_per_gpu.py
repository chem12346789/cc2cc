#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import shlex
import subprocess
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
TEST_SCRIPT_DIR = ROOT / "test_script"
TMP_DIR = TEST_SCRIPT_DIR / "tmp"
USABLE_STATES = {"idle", "mix", "mixed"}


def run_cmd(command: list[str], *, debug: bool = False, timeout_sec: int | None = None) -> str:
    if debug:
        print("[DEBUG] RUN:", " ".join(map(shlex.quote, command)))
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    if debug and result.stdout.strip():
        print(f"[DEBUG] STDOUT:\n{result.stdout.strip()}")
    if debug and result.stderr.strip():
        print(f"[DEBUG] STDERR:\n{result.stderr.strip()}")
    if result.returncode != 0:
        joined = " ".join(map(shlex.quote, command))
        raise RuntimeError(
            f"Command failed ({result.returncode}): {joined}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


def normalize_node(node_text: str) -> str:
    return node_text.strip().rstrip(",").replace("*", "").replace("~", "").replace("+", "").split()[0]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip(".-") or "run"


def resolve_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (TEST_SCRIPT_DIR, ROOT, Path.cwd()):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (ROOT / candidate).resolve()


def read_sbatch_field(script_text: str, field_keys: Iterable[str]) -> str | None:
    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("#SBATCH"):
            continue
        body = line[len("#SBATCH") :].strip()
        for key in field_keys:
            if body.startswith(f"{key}="):
                return body.split("=", 1)[1].strip()
            if body.startswith(f"{key} "):
                return body.split(None, 1)[1].strip()
    return None


def parse_array_ids(array_spec: str) -> list[int]:
    body = array_spec.strip().split("%", 1)[0]
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]

    ids: set[int] = set()
    for token in (segment.strip() for segment in body.split(",") if segment.strip()):
        match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)(?::(\d+))?", token)
        if match:
            start = int(match[1])
            end = int(match[2])
            step = int(match[3] or 1)
            if step <= 0:
                raise ValueError(f"Invalid array step in token: {token}")
            direction = 1 if start <= end else -1
            ids.update(range(start, end + direction, step * direction))
            continue
        if re.fullmatch(r"-?\d+", token):
            ids.add(int(token))
            continue
        raise ValueError(f"Unsupported array token: {token}")

    if not ids:
        raise ValueError(f"No task IDs parsed from array spec: {array_spec}")
    return sorted(ids)


def parse_array_limit(array_spec: str) -> int | None:
    if not (match := re.search(r"%\s*(\d+)\s*$", array_spec.strip())):
        return None
    return limit if (limit := int(match[1])) > 0 else None


def expand_hostnames(expression: str, *, debug: bool = False) -> list[str]:
    if not expression:
        return []
    try:
        rows = run_cmd(["scontrol", "show", "hostnames", expression], debug=debug)
        return [normalize_node(row) for row in rows.splitlines() if row.strip()]
    except RuntimeError:
        return [normalize_node(token) for token in expression.split(",") if token.strip()]


def get_partition_nodes(partition: str, *, states: set[str] | None, debug: bool = False) -> list[str]:
    rows = run_cmd(["sinfo", "-h", "-N", "-p", partition, "-o", "%n|%t"], debug=debug)
    nodes: list[str] = []
    for row in (line.strip() for line in rows.splitlines() if line.strip()):
        node, _, state = row.partition("|")
        if states is None or state.strip().lower().rstrip("*~+#") in states:
            nodes.extend(expand_hostnames(normalize_node(node), debug=debug))
    unique_nodes = list(dict.fromkeys(nodes))
    if not unique_nodes:
        raise RuntimeError(f"No nodes found in partition {partition!r}")
    return unique_nodes


def extract_load_model_arg(script_text: str, key: str) -> str | None:
    for raw_line in script_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "load_model_args" not in line or "--load" not in line:
            continue
        tokens = shlex.split(line, comments=True)
        for value in [
            token.split("=", 1)[1]
            for token in tokens
            if token.startswith("load_model_args=")
        ] or [line]:
            value_tokens = shlex.split(value, comments=True)
            for idx, token in enumerate(value_tokens):
                if token == f"--{key}" and idx + 1 < len(value_tokens):
                    return value_tokens[idx + 1]
                if token.startswith(f"--{key}="):
                    return token.split("=", 1)[1]
    return None


def parse_molecule_names(script_text: str) -> list[str]:
    if not (match := re.search(r"name_mol_input_list\s*=\s*\((.*?)\)", script_text, re.S)):
        return []
    names: list[str] = []
    for value in shlex.split(match.group(1), comments=True):
        item = value.strip().strip("\"'")
        if item.startswith("molecule_"):
            item = item[len("molecule_") :]
        names.append(safe_name(item))
    return names


def split_by_time(task_ids: list[int], runtime_by_task: dict[int, float], bucket_count: int) -> list[list[int]]:
    missing = [task_id for task_id in task_ids if task_id not in runtime_by_task]
    if missing:
        preview = ",".join(map(str, missing[:20]))
        raise ValueError(f"Missing runtime estimate(s): {preview}")

    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    loads = [0.0] * bucket_count
    for task_id in sorted(task_ids, key=lambda tid: (runtime_by_task[tid], tid), reverse=True):
        target = min(range(bucket_count), key=lambda idx: (loads[idx], len(buckets[idx]), idx))
        buckets[target].append(task_id)
        loads[target] += runtime_by_task[task_id]

    order = {task_id: i for i, task_id in enumerate(task_ids)}
    for bucket in buckets:
        bucket.sort(key=lambda task_id: (runtime_by_task[task_id], order[task_id]))
    return buckets


def clean_tmp(*, debug: bool = False) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for path in TMP_DIR.glob("gpu_probe_*"):
        with suppress(OSError):
            path.unlink()
            if debug:
                print(f"[DEBUG] removed old tmp: {path}")


def make_log_paths(script_path: Path, script_text: str) -> tuple[str, str]:
    load = safe_name(extract_load_model_arg(script_text, "load") or "no-load")
    epoch = safe_name(extract_load_model_arg(script_text, "load_epoch") or "no-epoch")
    log_dir = Path("log") / safe_name(script_path.stem) / load / epoch
    log_dir.mkdir(parents=True, exist_ok=True)
    base = str(log_dir / "%x-a%a")
    print(f"[INFO] Log path={base}")
    return f"{base}.out", f"{base}.err"


def wait_probe_done(job_id: str, timeout_sec: int) -> bool:
    deadline = time.monotonic() + max(1, timeout_sec)
    while time.monotonic() < deadline:
        result = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], text=True, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return True
        time.sleep(1)
    return False


def probe_node_gpus(node: str, partition: str, min_free_gb: float, max_power: float | None, timeout: int, debug: bool, nodes: str | None, ntasks_per_node: str | None, cpus_per_task: str | None) -> tuple[str, list[int]]:
    tag = uuid.uuid4().hex[:10]
    out_file = TMP_DIR / f"gpu_probe_{node}_{tag}.out"
    err_file = TMP_DIR / f"gpu_probe_{node}_{tag}.err"
    cmd = [
        "sbatch",
        "--parsable",
        "--partition",
        partition,
        "--nodelist",
        node,
        "--time",
        "00:00:30",
        "--job-name",
        "probe-gpu-mem",
        "--output",
        str(out_file),
        "--error",
        str(err_file),
        "--wrap",
        "nvidia-smi --query-gpu=memory.free,power.draw,index --format=csv,noheader,nounits",
    ]
    cmd += ["--nodes", nodes] if nodes else ["--nodes", "1"]
    cmd += ["--ntasks-per-node", ntasks_per_node] if ntasks_per_node else ["--ntasks", "1"]
    if cpus_per_task:
        cmd += ["--cpus-per-task", cpus_per_task]

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    text = ""
    job_id: str | None = None
    try:
        job_id = parse_job_id(run_cmd(cmd, debug=debug))
        if wait_probe_done(job_id, timeout) and out_file.exists():
            text = out_file.read_text().strip()
    except RuntimeError:
        text = ""
    finally:
        if job_id is not None:
            subprocess.run(["scancel", job_id], text=True, capture_output=True, check=False)
        for path in (out_file, err_file):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    min_mib = int(min_free_gb * 1024)
    usable: list[int] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")[:3]]
        if len(parts) < 3:
            continue
        free_mib = int(parts[0].split()[0])
        power = float(parts[1].split()[0])
        gpu_index = int(parts[2].split()[0])
        if free_mib >= min_mib and (max_power is None or power <= max_power):
            usable.append(gpu_index)
    return node, sorted(set(usable))


def parse_job_id(text: str) -> str:
    match = re.search(r"\b(\d+)\b", text)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from: {text!r}")
    return match[1]


def build_submit_command(
    partition: str,
    task_id: int,
    cpus_per_task: int | str,
    gpu_index: int | None,
    job_name: str,
    dependency: str | None,
    exclude: list[str],
    output_log_path: str | None,
    error_log_path: str | None,
    script_path: Path,
) -> list[str]:
    command = ["sbatch", "--parsable", "--partition", partition, "--array", str(task_id)]
    command += ["--cpus-per-task", str(cpus_per_task)]
    if gpu_index is not None:
        command += ["--export", f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_index}"]
    command += ["--job-name", job_name]
    if output_log_path and error_log_path:
        command += ["--output", output_log_path, "--error", error_log_path]
    if dependency:
        command += [f"--dependency={dependency}"]
    if exclude:
        command += ["--exclude", ",".join(exclude)]
    return command + [str(script_path)]


def submit_with_optional_exclude_retry(command: list[str], exclude: list[str], debug: bool) -> str:
    try:
        return parse_job_id(run_cmd(command, debug=debug))
    except RuntimeError as err:
        if "Invalid node name specified" not in str(err) or not exclude:
            raise
    retry_command: list[str] = []
    command_iter = iter(command)
    for token in command_iter:
        if token == "--exclude":
            next(command_iter, None)
            continue
        retry_command.append(token)
    return parse_job_id(run_cmd(retry_command, debug=debug))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="test_direct.bash")
    parser.add_argument("--time-array", required=True)
    parser.add_argument("--array-concurrency", type=int, default=1)
    parser.add_argument("--min-free-memory", type=float, default=15.0)
    parser.add_argument("--max-gpu-power", type=float, default=None)
    parser.add_argument("--probe-timeout-sec", type=int, default=10)
    parser.add_argument("--probe-workers", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    script_path = resolve_path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    script_text = script_path.read_text()
    clean_tmp(debug=args.debug)

    partition = read_sbatch_field(script_text, ["-p", "--partition"])
    array_spec = read_sbatch_field(script_text, ["-a", "--array"])
    if not partition or not array_spec:
        raise RuntimeError("Partition/array not found in script")

    task_ids = parse_array_ids(array_spec)
    slot_limit = max(1, parse_array_limit(array_spec) or args.array_concurrency)
    output_log_path, error_log_path = make_log_paths(script_path, script_text)
    excluded_nodes = set(expand_hostnames(read_sbatch_field(script_text, ["-x", "--exclude"]) or "", debug=args.debug))

    probe_nodes = get_partition_nodes(partition, states=USABLE_STATES, debug=args.debug)
    submission_pool = get_partition_nodes(partition, states=None, debug=args.debug)
    if excluded_nodes:
        probe_nodes = [node for node in probe_nodes if node not in excluded_nodes]
        submission_pool = list(dict.fromkeys([*submission_pool, *excluded_nodes]))
    if args.max_nodes > 0:
        probe_nodes = probe_nodes[: args.max_nodes]

    script_nodes = read_sbatch_field(script_text, ["-N", "--nodes"])
    script_ntasks_per_node = read_sbatch_field(script_text, ["--ntasks-per-node"])
    script_cpus_per_task = read_sbatch_field(script_text, ["-c", "--cpus-per-task"])

    if args.cpu_only:
        slots = [(expand_hostnames(node, debug=args.debug)[0], None) for node in probe_nodes[:slot_limit]]
    else:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.probe_workers)) as pool:
            tasks = [
                pool.submit(
                    probe_node_gpus,
                    node,
                    partition,
                    args.min_free_memory,
                    args.max_gpu_power,
                    args.probe_timeout_sec,
                    args.debug,
                    script_nodes,
                    script_ntasks_per_node,
                    script_cpus_per_task,
                )
                for node in probe_nodes
            ]
            slots: list[tuple[str, int | None]] = []
            for future in futures.as_completed(tasks):
                node, gpu_ids = future.result()
                hosts = expand_hostnames(node, debug=args.debug)
                if not hosts:
                    continue
                slots.extend((hosts[0], gpu_id) for gpu_id in gpu_ids)
                if len(slots) >= slot_limit:
                    break
            slots = slots[:slot_limit]

    if not slots:
        print("[WARN] No usable slots found. Nothing submitted.")
        return 0

    with resolve_path(args.time_array).open() as file:
        cfg = json.load(file)

    runtime_by_task = {task_id: float(runtime) for task_id, runtime in zip(task_ids, cfg["time_array"])}
    cpus_per_task_by_task = cfg["cpus_per_task"]
    task_buckets = split_by_time(task_ids, runtime_by_task, len(slots))
    assignment = {
        task_id: (bucket_idx + 1, slots[bucket_idx][0], slots[bucket_idx][1])
        for bucket_idx, bucket in enumerate(task_buckets)
        for task_id in bucket
    }

    model_name = safe_name(extract_load_model_arg(script_text, "load") or "no-load")
    molecule_names = parse_molecule_names(script_text)

    prev_job_by_bucket: dict[int, str] = {}
    submitted: list[tuple[int, int, str]] = []
    for task_id in task_ids:
        if runtime_by_task[task_id] == 0:
            continue
        bucket, node_name, gpu_index = assignment[task_id]
        dependency = f"afterany:{prev_job_by_bucket[bucket]}" if bucket in prev_job_by_bucket else None
        molecule_name = molecule_names[task_id] if 0 <= task_id < len(molecule_names) else f"task-{task_id}"
        exclude = [] if args.cpu_only else [node for node in submission_pool if node != node_name]
        command = build_submit_command(
            partition,
            task_id,
            cpus_per_task_by_task[task_id],
            gpu_index,
            safe_name(f"{model_name}-{molecule_name}"),
            dependency,
            exclude,
            output_log_path,
            error_log_path,
            script_path,
        )
        job_id = submit_with_optional_exclude_retry(command, exclude, args.debug)
        prev_job_by_bucket[bucket] = job_id
        submitted.append((task_id, bucket, job_id))

    by_bucket_jobs: dict[int, list[str]] = {}
    by_bucket_tasks: dict[int, list[str]] = {}
    for task_id, bucket, job_id in submitted:
        by_bucket_jobs.setdefault(bucket, []).append(job_id)
        by_bucket_tasks.setdefault(bucket, []).append(str(task_id))

    print("[PID-SUMMARY]", " ".join(" ".join(ids) for _, ids in sorted(by_bucket_jobs.items())))
    print("[SLURM_ARRAY_TASK_ID]", " ".join(f"{bucket}:{','.join(ids)}" for bucket, ids in sorted(by_bucket_tasks.items())))
    print(f"[DONE] submissions={len(submitted)}, tasks={len(task_ids)}, slots={len(slots)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
