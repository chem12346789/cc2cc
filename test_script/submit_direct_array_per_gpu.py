#!/usr/bin/env python3
"""Submit Slurm array jobs by splitting tasks across GPUs with enough free memory.

Behavior:
1) Parse partition + array from the target sbatch script.
2) Probe each node via a tiny `sbatch --wait` job (no srun) that runs nvidia-smi.
3) Keep GPU slots with free memory >= threshold.
4) Split array task IDs across slots and submit per-node array jobs.

Temp files are stored in: test_script/tmp
Old probe files are cleaned at startup.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp"


def run(cmd: list[str], timeout_sec: int | None = None, debug: bool = False) -> str:
    if debug:
        print(f"[DEBUG] RUN: {' '.join(map(shlex.quote, cmd))}")
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
    if debug:
        if p.stdout.strip():
            print(f"[DEBUG] STDOUT:\n{p.stdout.strip()}")
        if p.stderr.strip():
            print(f"[DEBUG] STDERR:\n{p.stderr.strip()}")
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(map(shlex.quote, cmd))}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def normalize_node_token(token: str) -> str:
    t = token.strip().rstrip(",")
    t = t.replace("*", "").replace("~", "").replace("+", "")
    return t.split()[0] if t else t


def expand_hostlist_expr(expr: str, debug: bool = False) -> set[str]:
    """Expand Slurm hostlist expression to concrete node names."""
    expr = expr.strip()
    if not expr:
        return set()
    try:
        out = run(["scontrol", "show", "hostnames", expr], debug=debug)
        return {normalize_node_token(x) for x in out.splitlines() if x.strip()}
    except Exception:
        # Fallback: treat as comma-separated plain hostnames
        return {
            normalize_node_token(x)
            for x in expr.split(",")
            if normalize_node_token(x)
        }


def parse_sbatch_field(text: str, keys: Iterable[str]) -> str | None:
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("#SBATCH"):
            continue
        body = line[len("#SBATCH") :].strip()
        for key in keys:
            if body.startswith(key + "="):
                return body.split("=", 1)[1].strip()
            if body.startswith(key + " "):
                return body.split(None, 1)[1].strip()
    return None


def expand_array_spec(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty array spec")

    spec = spec.split("%", 1)[0]
    if spec.startswith("[") and spec.endswith("]"):
        spec = spec[1:-1]

    out: set[int] = set()
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        m = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)(?::(\d+))?", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            step = int(m.group(3) or "1")
            if step <= 0:
                raise ValueError(f"Invalid step in token: {tok}")
            if a <= b:
                out.update(range(a, b + 1, step))
            else:
                out.update(range(a, b - 1, -step))
            continue
        if re.fullmatch(r"-?\d+", tok):
            out.add(int(tok))
            continue
        raise ValueError(f"Unsupported array token: {tok}")

    if not out:
        raise ValueError(f"No task IDs parsed from array spec: {spec}")
    return sorted(out)


def parse_array_concurrency(spec: str) -> int | None:
    """Return %N from array spec (e.g. 1-23%2 -> 2), else None."""
    m = re.search(r"%\s*(\d+)\s*$", spec.strip())
    if not m:
        return None
    v = int(m.group(1))
    return v if v > 0 else None


def compress_array_ids(task_ids: list[int]) -> str:
    task_ids = sorted(set(task_ids))
    if not task_ids:
        raise ValueError("Empty task list")

    ranges: list[str] = []
    start = prev = task_ids[0]
    for x in task_ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def format_array_ids_in_order(task_ids: list[int]) -> str:
    """Format array IDs without reordering them.

    Consecutive ascending runs are compressed, but separated runs keep the
    incoming order. This keeps the submitted array string aligned with the
    original task order after tasks have been assigned to balanced buckets.
    """
    seen: set[int] = set()
    ordered_ids: list[int] = []
    for tid in task_ids:
        if tid not in seen:
            ordered_ids.append(tid)
            seen.add(tid)
    if not ordered_ids:
        raise ValueError("Empty task list")

    ranges: list[str] = []
    start = prev = ordered_ids[0]
    for x in ordered_ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def preview_list(items: Iterable[object], max_items: int = 20) -> str:
    values = [str(x) for x in items]
    if len(values) <= max_items:
        return ",".join(values)
    return ",".join(values[:max_items]) + f",...(+{len(values) - max_items})"


def format_task_time_details(
    task_ids: list[int], estimates: dict[int, float], max_items: int = 80
) -> str:
    details = [f"{tid}:{estimates[tid]:.2f}" for tid in task_ids]
    if len(details) <= max_items:
        return " ".join(details)
    return " ".join(details[:max_items]) + f" ...(+{len(details) - max_items})"


def load_task_time_estimates(path: Path, offset: int = 0) -> dict[int, float]:
    """Load task runtime estimates.

    Supported formats:
    - One list of numeric values separated by comma and/or whitespace.
      With offset=0, value position i maps to task id i.
    - Two numeric columns per non-comment line: task_id,time.

    Lines may contain comments after '#'.
    """
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        row: list[float] = []
        bad_tokens: list[str] = []
        for tok in re.split(r"[\s,]+", line):
            if not tok:
                continue
            try:
                row.append(float(tok))
            except ValueError:
                bad_tokens.append(tok)
        if bad_tokens:
            if not row:
                # Allow simple CSV headers such as "task_id,time".
                continue
            raise ValueError(f"Non-numeric token in {path}: {bad_tokens[0]!r}")
        if row:
            rows.append(row)

    if not rows:
        raise ValueError(f"No numeric runtime estimates found in {path}")

    estimates: dict[int, float] = {}
    if len(rows) > 1 and all(len(r) == 2 and float(r[0]).is_integer() for r in rows):
        for task_id_float, runtime in rows:
            task_id = int(task_id_float)
            if task_id in estimates:
                raise ValueError(f"Duplicate task id {task_id} in {path}")
            if not math.isfinite(runtime) or runtime < 0:
                raise ValueError(f"Invalid runtime for task id {task_id}: {runtime}")
            estimates[task_id] = runtime
    else:
        values = [v for row in rows for v in row]
        for i, runtime in enumerate(values):
            if not math.isfinite(runtime) or runtime < 0:
                raise ValueError(f"Invalid runtime at position {i}: {runtime}")
            estimates[offset + i] = runtime

    return estimates


def split_tasks_round_robin(task_ids: list[int], n_buckets: int) -> tuple[list[list[int]], list[float]]:
    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    for i, tid in enumerate(task_ids):
        buckets[i % n_buckets].append(tid)
    return buckets, [float(len(b)) for b in buckets]


def split_tasks_by_estimated_time(
    task_ids: list[int], estimates: dict[int, float], n_buckets: int
) -> tuple[list[list[int]], list[float]]:
    """Greedily balance total estimated runtime across buckets.

    This uses the Longest Processing Time (LPT) heuristic: assign the longest
    remaining task to the currently lightest bucket.
    """
    missing = [tid for tid in task_ids if tid not in estimates]
    if missing:
        preview = ",".join(map(str, missing[:20]))
        if len(missing) > 20:
            preview += ",..."
        raise ValueError(f"Missing runtime estimate(s) for task id(s): {preview}")

    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    loads = [0.0 for _ in range(n_buckets)]
    for tid in sorted(task_ids, key=lambda x: (estimates[x], x), reverse=True):
        bucket_idx = min(range(n_buckets), key=lambda i: (loads[i], len(buckets[i]), i))
        buckets[bucket_idx].append(tid)
        loads[bucket_idx] += estimates[tid]

    # Keep each submitted array in the original task order after balancing.
    # The LPT ordering is used only to choose the bucket, not to submit jobs.
    order_index = {tid: i for i, tid in enumerate(task_ids)}
    for bucket in buckets:
        bucket.sort(key=lambda tid: order_index[tid])

    return buckets, loads


def resolve_relative_to_root_or_cwd(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    candidate1 = (ROOT / path).resolve()
    candidate2 = path.resolve()
    return candidate1 if candidate1.exists() else candidate2


def partition_nodes(partition: str, debug: bool = False) -> list[str]:
    out = run(
        ["sinfo", "-h", "-N", "-p", partition, "-o", "%n|%t"],
        debug=debug,
    )
    raw_nodes = [n.strip() for n in out.splitlines() if n.strip()]
    nodes: list[str] = []
    for item in raw_nodes:
        parts = item.split("|", 1)
        node_tok = normalize_node_token(parts[0])
        state_tok = parts[1].strip().lower() if len(parts) > 1 else ""
        # Keep only clearly allocatable states; skip special suffix states like idle~.
        if state_tok not in {"idle", "mix", "mixed"}:
            continue
        item = node_tok
        if not item:
            continue
        # Expand hostlist safely (handles patterns like gpu[01-04]).
        try:
            expanded = run(["scontrol", "show", "hostnames", item], debug=debug)
            nodes.extend(
                [normalize_node_token(x) for x in expanded.splitlines() if x.strip()]
            )
        except Exception:
            # Fallback to raw token.
            nodes.append(item)
    # De-duplicate while preserving order.
    nodes = list(dict.fromkeys(nodes))
    if not nodes:
        raise RuntimeError(f"No nodes found in partition '{partition}'")
    return nodes


def clean_old_probe_files(tmp_dir: Path, debug: bool = False) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in tmp_dir.glob("gpu_probe_*"):
        try:
            p.unlink()
            if debug:
                print(f"[DEBUG] removed old tmp: {p}")
        except Exception:
            pass


def probe_node_via_sbatch(
    node: str,
    partition: str,
    min_free_gb: float,
    max_gpu_power: float | None,
    probe_timeout_sec: int,
    tmp_dir: Path,
    debug: bool,
) -> tuple[str, list[int]]:
    probe_id = uuid.uuid4().hex[:10]
    out_file = tmp_dir / f"gpu_probe_{node}_{probe_id}.out"
    err_file = tmp_dir / f"gpu_probe_{node}_{probe_id}.err"

    cmd = [
        "timeout",
        f"{probe_timeout_sec}s",
        "sbatch",
        "--parsable",
        "--wait",
        "--partition",
        partition,
        "--nodelist",
        node,
        "--nodes",
        "1",
        "--ntasks",
        "1",
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

    try:
        meta = run(cmd, timeout_sec=probe_timeout_sec + 3, debug=debug)
        if debug:
            print(f"[DEBUG] probe submit/wait result: {meta}")
    except Exception as e:
        if debug:
            print(f"[DEBUG] Probe failed on {node}: {e}")
        return node, []

    try:
        probe_stdout = out_file.read_text().strip()
    except Exception:
        probe_stdout = ""
    try:
        probe_stderr = err_file.read_text().strip()
    except Exception:
        probe_stderr = ""

    if debug:
        if probe_stdout:
            print(f"[DEBUG] PROBE STDOUT ({node}):\n{probe_stdout}")
        if probe_stderr:
            print(f"[DEBUG] PROBE STDERR ({node}):\n{probe_stderr}")

    eligible_gpu_indices: list[int] = []
    min_free_mib = int(min_free_gb * 1024)
    for line in probe_stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            parts = [x.strip() for x in s.split(",")]
            free_mib = int(parts[0].split()[0])
            gpu_idx = int(parts[2].split()[0])
            if max_gpu_power is None:
                power_ok = True
            else:
                # nvidia-smi can return "N/A" for power.draw
                power_ok = False
                if len(parts) > 1:
                    ptxt = parts[1]
                    try:
                        power_ok = float(ptxt.split()[0]) <= max_gpu_power
                    except ValueError:
                        power_ok = False
            if free_mib >= min_free_mib and power_ok:
                eligible_gpu_indices.append(gpu_idx)
        except (ValueError, IndexError):
            pass

    for p in (out_file, err_file):
        try:
            p.unlink()
        except Exception:
            pass

    return node, sorted(set(eligible_gpu_indices))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--script",
        default="test_direct.bash",
        help="Slurm bash script path (supports relative path; relative paths are resolved under test_script/ first)",
    )
    ap.add_argument("--partition", default=None, help="Override partition")
    ap.add_argument("--array", default=None, help="Override array spec")
    ap.add_argument(
        "--min-free-memory",
        type=float,
        default=15.0,
        help="GPU free memory threshold in GB",
    )
    ap.add_argument(
        "--max-gpu-power", type=float, default=None, help="Max GPU power draw in W"
    )
    ap.add_argument("--array-concurrency", type=int, default=1)
    ap.add_argument(
        "--time-array",
        default=None,
        help=(
            "Runtime estimate file for time-balanced task splitting. "
            "If omitted, test_script/task_times_0_54.csv is used when it exists "
            "and covers all selected array task IDs. Use --no-time-balance to disable."
        ),
    )
    ap.add_argument(
        "--time-array-offset",
        type=int,
        default=0,
        help=(
            "For a one-row/list time array, map value position i to task id "
            "offset+i. Ignored for explicit task_id,time two-column files."
        ),
    )
    ap.add_argument(
        "--no-time-balance",
        action="store_true",
        help="Disable runtime-estimate balancing and use round-robin splitting.",
    )
    ap.add_argument("--probe-timeout-sec", type=int, default=60)
    ap.add_argument("--probe-workers", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=0, help="0 means no limit")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_script = Path(args.script).expanduser()
    if raw_script.is_absolute():
        script_path = raw_script
    else:
        # Prefer path relative to this python file directory: test_script/
        candidate1 = (ROOT / raw_script).resolve()
        candidate2 = raw_script.resolve()
        script_path = candidate1 if candidate1.exists() else candidate2
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    clean_old_probe_files(TMP_DIR, debug=args.debug)

    text = script_path.read_text()
    partition = args.partition or parse_sbatch_field(text, ["-p", "--partition"])
    if not partition:
        raise RuntimeError("Partition not found in script. Use --partition.")

    array_spec = args.array or parse_sbatch_field(text, ["-a", "--array"])
    if not array_spec:
        raise RuntimeError("Array spec not found in script. Use --array.")
    exclude_spec = parse_sbatch_field(text, ["-x", "--exclude"]) or ""
    original_array_concurrency = parse_array_concurrency(array_spec)

    task_ids = expand_array_spec(array_spec)
    print(f"[INFO] Submit helper script: {Path(__file__).resolve()}")
    print(f"[INFO] Target sbatch script: {script_path}")
    print(f"[INFO] Partition: {partition}")
    print(f"[INFO] Array spec: {array_spec}")
    print(
        f"[INFO] Parsed {len(task_ids)} task ID(s): "
        f"{compress_array_ids(task_ids)}"
    )
    if original_array_concurrency is not None:
        print(f"[INFO] Array concurrency from script: %{original_array_concurrency}")
    else:
        print(f"[INFO] No array concurrency in script; fallback limit: {args.array_concurrency}")
    if exclude_spec:
        print(f"[INFO] Exclude spec from script: {exclude_spec}")

    nodes = partition_nodes(partition, debug=args.debug)
    if exclude_spec:
        excluded_nodes = expand_hostlist_expr(exclude_spec, debug=args.debug)
        if excluded_nodes:
            nodes = [n for n in nodes if n not in excluded_nodes]
            print(f"[INFO] Excluding {len(excluded_nodes)} node(s) from script --exclude")
    if args.max_nodes > 0:
        nodes = nodes[: args.max_nodes]
        print(f"[INFO] Applied --max-nodes={args.max_nodes}")
    print(f"[INFO] Candidate node count: {len(nodes)}")
    print(f"[INFO] Candidate node preview: {preview_list(nodes)}")

    print(
        f"[INFO] Scanning partition '{partition}' for GPUs with free memory >= {args.min_free_memory} GB..."
    )
    if args.max_gpu_power is not None:
        print(f"[INFO] Additional filter: GPU power draw <= {args.max_gpu_power} W")
    print(f"[INFO] Probing {len(nodes)} node(s) with {args.probe_workers} worker(s)...")

    slots: list[tuple[str, int]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.probe_workers)) as ex:
        futs = [
            ex.submit(
                probe_node_via_sbatch,
                n,
                partition,
                args.min_free_memory,
                args.max_gpu_power,
                args.probe_timeout_sec,
                TMP_DIR,
                args.debug,
            )
            for n in nodes
        ]
        done = 0
        for fut in cf.as_completed(futs):
            node, gpu_indices = fut.result()
            done += 1
            if gpu_indices:
                print(f"  - {node}: eligible GPU(s) = {gpu_indices}")
                slots.extend([(node, idx) for idx in gpu_indices])
            if done % 20 == 0 or done == len(nodes):
                print(f"[INFO] Probe progress: {done}/{len(nodes)}")

    if not slots:
        print("[WARN] No eligible GPUs found. Nothing submitted.")
        return 0
    print(f"[INFO] Eligible GPU slot count before validation: {len(slots)}")
    print(f"[INFO] Eligible GPU slot preview: {preview_list(f'{n}:gpu{g}' for n, g in slots)}")

    # Validate node names once more to avoid sbatch "Invalid node name specified".
    valid_slots: list[tuple[str, int]] = []
    for node, gpu_idx in slots:
        try:
            chk = run(["scontrol", "show", "hostnames", node], debug=args.debug)
            if chk.strip():
                valid_slots.append((node, gpu_idx))
        except Exception:
            if args.debug:
                print(f"[DEBUG] drop invalid node token: {node}")
    slots = valid_slots
    if not slots:
        print("[WARN] No valid eligible GPUs found after node validation. Nothing submitted.")
        return 0
    print(f"[INFO] Valid GPU slot count after validation: {len(slots)}")
    print(f"[INFO] Valid GPU slot preview: {preview_list(f'{n}:gpu{g}' for n, g in slots)}")

    # Limit total running jobs using original '#SBATCH -a ...%N' setting.
    # If original spec has no %N, fallback to --array-concurrency.
    global_limit = max(1, original_array_concurrency or args.array_concurrency)
    if len(slots) > global_limit:
        print(
            f"[INFO] Trimming valid slots from {len(slots)} to {global_limit} "
            "because of the global running-job limit."
        )
    slots = slots[: min(len(slots), global_limit)]
    print(
        f"[INFO] Global running-job limit = {global_limit}; using {len(slots)} GPU slot(s)"
    )
    print(f"[INFO] Final GPU slot list: {preview_list(f'{n}:gpu{g}' for n, g in slots)}")

    buckets: list[list[int]]
    bucket_loads: list[float]
    time_array_path: Path | None = None
    estimates: dict[int, float] = {}
    explicit_time_array = args.time_array is not None
    if not args.no_time_balance:
        if args.time_array is None:
            auto_path = ROOT / "task_times_0_54.csv"
            if auto_path.exists():
                time_array_path = auto_path
                print(f"[INFO] Auto runtime estimate file detected: {time_array_path}")
            else:
                print(f"[INFO] Auto runtime estimate file not found: {auto_path}")
        elif args.time_array.strip().lower() not in {"", "none", "off", "false", "0"}:
            time_array_path = resolve_relative_to_root_or_cwd(args.time_array)
            print(f"[INFO] Runtime estimate file requested: {time_array_path}")
        else:
            print("[INFO] Runtime estimate file disabled by --time-array value.")
    else:
        print("[INFO] Time balancing disabled by --no-time-balance.")

    if time_array_path is not None:
        if not time_array_path.exists():
            if explicit_time_array:
                raise FileNotFoundError(time_array_path)
            print(f"[WARN] Runtime estimate file not found: {time_array_path}")
            time_array_path = None

    if time_array_path is not None:
        estimates = load_task_time_estimates(time_array_path, offset=args.time_array_offset)
        missing = [tid for tid in task_ids if tid not in estimates]
        if missing and not explicit_time_array:
            print(
                "[WARN] Auto runtime estimate file does not cover all selected "
                "task IDs; falling back to round-robin splitting."
            )
            if args.debug:
                print(f"[DEBUG] missing task IDs: {missing}")
            time_array_path = None
        elif missing:
            preview = ",".join(map(str, missing[:20]))
            if len(missing) > 20:
                preview += ",..."
            raise RuntimeError(
                f"Runtime estimate file {time_array_path} is missing task ID(s): {preview}"
            )
        if time_array_path is not None:
            selected_times = [estimates[tid] for tid in task_ids]
            print(f"[INFO] Loaded {len(estimates)} runtime estimate(s).")
            print(
                "[INFO] Selected task runtime stats: "
                f"total={sum(selected_times):.2f}, "
                f"min={min(selected_times):.2f}, "
                f"max={max(selected_times):.2f}, "
                f"avg={sum(selected_times) / len(selected_times):.2f}"
            )
            slowest = sorted(
                task_ids, key=lambda tid: (estimates[tid], tid), reverse=True
            )[:10]
            print(
                "[INFO] Slowest selected task(s): "
                + " ".join(f"{tid}:{estimates[tid]:.2f}" for tid in slowest)
            )

    if time_array_path is not None:
        buckets, bucket_loads = split_tasks_by_estimated_time(task_ids, estimates, len(slots))
        nonempty_loads = [load for load, bucket in zip(bucket_loads, buckets) if bucket]
        if nonempty_loads:
            print(
                f"[INFO] Time-balanced splitting using {time_array_path} "
                f"(offset={args.time_array_offset}): "
                f"min/max estimated load = {min(nonempty_loads):.2f}/{max(nonempty_loads):.2f}"
            )
    else:
        buckets, bucket_loads = split_tasks_round_robin(task_ids, len(slots))
        print("[INFO] Using round-robin task splitting.")

    for idx, ((node, gpu_idx), ids, load) in enumerate(
        zip(slots, buckets, bucket_loads), start=1
    ):
        if not ids:
            continue
        label = "est_load" if time_array_path is not None else "task_count"
        print(
            f"[INFO] bucket {idx}: node={node}, gpu={gpu_idx}, "
            f"tasks={len(ids)}, {label}={load:.2f}, ids={compress_array_ids(ids)}"
        )
        if time_array_path is not None:
            print(f"[INFO] bucket {idx} task:time detail: {format_task_time_details(ids, estimates)}")

    submitted = 0
    submit_pool_nodes = list(dict.fromkeys(nodes))
    for (node, gpu_idx), ids in zip(slots, buckets):
        if not ids:
            continue
        node = normalize_node_token(node)
        if not node:
            continue
        try:
            canon = run(["scontrol", "show", "hostnames", node], debug=args.debug)
            node_submit = (
                normalize_node_token(canon.splitlines()[0])
                if canon.strip()
                else normalize_node_token(node)
            )
        except Exception:
            node_submit = normalize_node_token(node)
        # one running task per submission; global limit enforced by number of submissions
        arr = f"{compress_array_ids(ids)}%1"
        print(
            f"[INFO] Preparing submission {submitted + 1}: "
            f"node={node_submit}, gpu={gpu_idx}, array={arr}, tasks={len(ids)}"
        )
        cmd = [
            "sbatch",
            "--partition",
            partition,
            "--export",
            f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_idx}",
            "--array",
            arr,
        ]
        # User preference: avoid --nodelist; emulate pinning by excluding other nodes.
        exclude_nodes = [n for n in submit_pool_nodes if n != node_submit]
        if exclude_nodes:
            cmd.extend(["--exclude", ",".join(exclude_nodes)])
        cmd.append(str(script_path))
        print("[SUBMIT]", " ".join(map(shlex.quote, cmd)))
        if not args.dry_run:
            try:
                out = run(cmd, debug=args.debug)
                print("        ", out)
            except Exception as e:
                if "Invalid node name specified" in str(e):
                    print(f"[WARN] Invalid node '{node_submit}', retry submit without node filter")
                    retry_cmd = [
                        "sbatch",
                        "--partition",
                        partition,
                        "--export",
                        f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_idx}",
                        "--array",
                        arr,
                        str(script_path),
                    ]
                    print("[SUBMIT-RETRY]", " ".join(map(shlex.quote, retry_cmd)))
                    out = run(retry_cmd, debug=args.debug)
                    print("        ", out)
                else:
                    raise
        submitted += 1

    print(f"[DONE] submissions={submitted}, tasks={len(task_ids)}, slots={len(slots)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
