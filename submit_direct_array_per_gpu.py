#!/usr/bin/env python3
"""Submit Slurm array tasks by splitting tasks across GPUs with enough free memory.

Behavior:
1) Parse partition + array from the target sbatch script.
2) Probe each node via a tiny `sbatch --wait` job (no srun) that runs nvidia-smi.
3) Keep GPU slots with free memory >= threshold.
4) Split array task IDs across slots.
5) Submit each task individually in original task order, with per-slot dependency
   chains so each GPU runs only one assigned task at a time.

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
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
TEST_SCRIPT_DIR = ROOT / "test_script"
TMP_DIR = TEST_SCRIPT_DIR / "tmp"


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


def parse_sbatch_job_id(output: str) -> str:
    """Extract the numeric Slurm job id from sbatch output."""
    text = output.strip()
    m = re.search(r"\b(\d+)\b", text)
    if not m:
        raise RuntimeError(f"Could not parse sbatch job id from output: {output!r}")
    return m.group(1)


def safe_log_name(text: str) -> str:
    """Return a filesystem-friendly log filename component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return safe.strip(".-") or "run"


def join_log_path(log_dir: str, filename: str) -> str:
    log_dir = log_dir.rstrip("/")
    return f"{log_dir}/{filename}" if log_dir and log_dir != "." else filename


def parse_load_model_name(script_text: str) -> str | None:
    """Parse the value after '--load' from an exported load_model_args line."""
    for raw in script_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "load_model_args" not in line or "--load" not in line:
            continue

        values: list[str] = []
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            tokens = []
        for tok in tokens:
            if tok.startswith("load_model_args="):
                values.append(tok.split("=", 1)[1])
        if not values:
            values.append(line)

        for value in values:
            try:
                value_tokens = shlex.split(value, comments=True, posix=True)
            except ValueError:
                value_tokens = value.split()
            for i, tok in enumerate(value_tokens):
                if tok == "--load" and i + 1 < len(value_tokens):
                    return value_tokens[i + 1]
                if tok.startswith("--load="):
                    return tok.split("=", 1)[1]

            m = re.search(r"--load(?:=|\s+)([^\s\"'#]+)", value)
            if m:
                return m.group(1)
    return None


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
            normalize_node_token(x) for x in expr.split(",") if normalize_node_token(x)
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


def format_array_ids(task_ids: Iterable[int], preserve_order: bool = False) -> str:
    """Format Slurm array IDs, optionally preserving incoming order."""
    if preserve_order:
        seen: set[int] = set()
        ordered_ids: list[int] = []
        for tid in task_ids:
            if tid not in seen:
                ordered_ids.append(tid)
                seen.add(tid)
    else:
        ordered_ids = sorted(set(task_ids))

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


def split_tasks_round_robin(
    task_ids: list[int], n_buckets: int
) -> tuple[list[list[int]], list[float]]:
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

    # Within each balanced bucket, run shorter estimated jobs before longer ones.
    # The descending LPT order above is used only to choose the bucket.
    order_index = {tid: i for i, tid in enumerate(task_ids)}
    for bucket in buckets:
        bucket.sort(key=lambda tid: (estimates[tid], order_index[tid]))

    return buckets, loads


def resolve_relative_to_root_or_cwd(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    for base in (TEST_SCRIPT_DIR, ROOT, Path.cwd()):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


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
    ap.add_argument(
        "--log-dir",
        default="log",
        help="Directory for sbatch stdout/stderr files when log override is enabled.",
    )
    ap.add_argument(
        "--log-prefix",
        default=None,
        help=(
            "Common log filename prefix for all jobs submitted by this Python run. "
            "Default: <target-script-stem>-<load-model>-<timestamp>."
        ),
    )
    ap.add_argument(
        "--no-log-override",
        action="store_true",
        help="Do not pass sbatch --output/--error; use #SBATCH -o/-e from the script.",
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_script = Path(args.script).expanduser()
    if raw_script.is_absolute():
        script_path = raw_script
    else:
        # Prefer path relative to test_script/, then repo root/current cwd.
        script_path = resolve_relative_to_root_or_cwd(str(raw_script))
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    text = script_path.read_text()

    log_output: str | None = None
    log_error: str | None = None
    if not args.no_log_override:
        load_model_name = parse_load_model_name(text) or "no-load"
        log_prefix = safe_log_name(
            args.log_prefix
            or f"{script_path.stem}-{load_model_name}-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:4]}"
        )
        log_dir_arg = args.log_dir.strip() or "."
        log_dir_path = Path(log_dir_arg).expanduser()
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_dir_for_sbatch = (
            str(log_dir_path) if log_dir_arg.startswith("~") else log_dir_arg
        )
        log_stem = f"{log_prefix}-a%a"
        log_output = join_log_path(log_dir_for_sbatch, f"{log_stem}.out")
        log_error = join_log_path(log_dir_for_sbatch, f"{log_stem}.err")
        print(f"[INFO] Parsed --load for log prefix: {load_model_name}")
        print(f"[INFO] Per-Python-run log prefix: {log_prefix}")
        print(f"[INFO] sbatch stdout override: {log_output}")
        print(f"[INFO] sbatch stderr override: {log_error}")
    else:
        print("[INFO] Log override disabled; using #SBATCH -o/-e from target script.")

    clean_old_probe_files(TMP_DIR, debug=args.debug)

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
    print(f"[INFO] Parsed {len(task_ids)} task ID(s): " f"{format_array_ids(task_ids)}")
    if original_array_concurrency is not None:
        print(f"[INFO] Array concurrency from script: %{original_array_concurrency}")
    else:
        print(
            f"[INFO] No array concurrency in script; fallback limit: {args.array_concurrency}"
        )
    if exclude_spec:
        print(f"[INFO] Exclude spec from script: {exclude_spec}")

    nodes = partition_nodes(partition, debug=args.debug)
    if exclude_spec:
        excluded_nodes = expand_hostlist_expr(exclude_spec, debug=args.debug)
        if excluded_nodes:
            nodes = [n for n in nodes if n not in excluded_nodes]
            print(
                f"[INFO] Excluding {len(excluded_nodes)} node(s) from script --exclude"
            )
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
    print(
        f"[INFO] Eligible GPU slot preview: {preview_list(f'{n}:gpu{g}' for n, g in slots)}"
    )

    # Validate and canonicalize node names once to avoid duplicate scontrol calls
    # and sbatch "Invalid node name specified" errors.
    valid_slots: list[tuple[str, int]] = []
    for node, gpu_idx in slots:
        node_norm = normalize_node_token(node)
        if not node_norm:
            continue
        try:
            chk = run(["scontrol", "show", "hostnames", node_norm], debug=args.debug)
            if chk.strip():
                valid_slots.append((normalize_node_token(chk.splitlines()[0]), gpu_idx))
        except Exception:
            if args.debug:
                print(f"[DEBUG] drop invalid node token: {node}")
    slots = valid_slots
    if not slots:
        print(
            "[WARN] No valid eligible GPUs found after node validation. Nothing submitted."
        )
        return 0
    print(f"[INFO] Valid GPU slot count after validation: {len(slots)}")
    print(
        f"[INFO] Valid GPU slot preview: {preview_list(f'{n}:gpu{g}' for n, g in slots)}"
    )

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
    print(
        f"[INFO] Final GPU slot list: {preview_list(f'{n}:gpu{g}' for n, g in slots)}"
    )

    buckets: list[list[int]]
    bucket_loads: list[float]
    time_array_path: Path | None = None
    estimates: dict[int, float] = {}
    explicit_time_array = args.time_array is not None
    if not args.no_time_balance:
        if args.time_array is None:
            auto_path = TEST_SCRIPT_DIR / "task_times_0_54.csv"
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
        estimates = load_task_time_estimates(
            time_array_path, offset=args.time_array_offset
        )
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
            skipped_zero_time_task_ids = [
                tid for tid in task_ids if estimates[tid] == 0
            ]
            if skipped_zero_time_task_ids:
                print(
                    "[INFO] Skipping task ID(s) with est_time == 0: "
                    f"{format_array_ids(skipped_zero_time_task_ids)}"
                )
                task_ids = [tid for tid in task_ids if estimates[tid] != 0]
                print(
                    f"[INFO] Remaining task ID(s) after zero-time skip: "
                    f"{len(task_ids)}"
                )
                if not task_ids:
                    print(
                        "[WARN] All selected tasks have est_time == 0. Nothing submitted."
                    )
                    return 0
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
        buckets, bucket_loads = split_tasks_by_estimated_time(
            task_ids, estimates, len(slots)
        )
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

    # Build a task -> balanced bucket assignment, then submit individual tasks
    # in the original array-task order.  Per-bucket dependency chains preserve
    # one-running-task-per-GPU behavior.
    task_assignment: dict[int, tuple[int, str, int, float]] = {}
    for bucket_idx, ((node_submit, gpu_idx), ids, load) in enumerate(
        zip(slots, buckets, bucket_loads), start=1
    ):
        if not ids:
            continue
        label = "est_load" if time_array_path is not None else "task_count"
        print(
            f"[INFO] bucket {bucket_idx}: node={node_submit}, gpu={gpu_idx}, "
            f"tasks={len(ids)}, {label}={load:.2f}, "
            f"ids={format_array_ids(ids, preserve_order=True)}"
        )
        if time_array_path is not None:
            print(
                f"[INFO] bucket {bucket_idx} task:time detail: "
                f"{format_task_time_details(ids, estimates)}"
            )
        for tid in ids:
            task_assignment[tid] = (bucket_idx, node_submit, gpu_idx, load)

    missing_assignment = [tid for tid in task_ids if tid not in task_assignment]
    if missing_assignment:
        raise RuntimeError(
            "Internal error: no balanced bucket assignment for task ID(s): "
            + preview_list(missing_assignment)
        )

    order_index = {tid: i for i, tid in enumerate(task_ids)}
    individual_plan = [(tid, *task_assignment[tid]) for tid in task_ids]
    if time_array_path is not None:
        individual_plan.sort(
            key=lambda item: (estimates[item[0]], order_index[item[0]])
        )
        order_msg = "ascending estimated runtime order (small est_time first)"
    else:
        order_msg = "original array task order because no est_time is available"
    print(
        "[INFO] Individual submission mode: submitting one sbatch job per "
        f"array task in {order_msg}."
    )
    print(
        "[INFO] Per-bucket dependencies use afterany:<previous_job_id>, so each "
        "GPU slot runs its rebalanced task chain sequentially."
    )

    submitted = 0
    submit_pool_nodes = list(dict.fromkeys(nodes))
    prev_job_id_by_bucket: dict[int, str] = {}
    dry_run_job_id_base = 900_000_000

    for (
        task_submit_idx,
        (tid, bucket_idx, node_submit, gpu_idx, bucket_load),
    ) in enumerate(individual_plan, start=1):
        if not node_submit:
            print(f"[WARN] Skip task {tid}: empty submit node for bucket {bucket_idx}")
            continue

        dependency_job_id = prev_job_id_by_bucket.get(bucket_idx)
        dependency_arg = (
            f"afterany:{dependency_job_id}" if dependency_job_id is not None else None
        )
        arr = str(tid)
        if time_array_path is not None:
            metric = (
                f"est_time={estimates[tid]:.2f}, " f"bucket_est_load={bucket_load:.2f}"
            )
        else:
            metric = f"bucket_task_count={bucket_load:.0f}"
        dep_text = dependency_arg if dependency_arg is not None else "none"
        print(
            f"[INFO] Preparing individual submission "
            f"{task_submit_idx}/{len(individual_plan)}: task_id={tid}, "
            f"bucket={bucket_idx}, node={node_submit}, gpu={gpu_idx}, "
            f"array={arr}, dependency={dep_text}, {metric}"
        )

        cmd = [
            "sbatch",
            "--parsable",
            "--partition",
            partition,
            "--export",
            f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_idx}",
            "--array",
            arr,
        ]
        if log_output is not None and log_error is not None:
            cmd.extend(["--output", log_output, "--error", log_error])
        if dependency_arg is not None:
            cmd.append(f"--dependency={dependency_arg}")
        # User preference: avoid --nodelist; emulate pinning by excluding other nodes.
        exclude_nodes = [n for n in submit_pool_nodes if n != node_submit]
        if exclude_nodes:
            cmd.extend(["--exclude", ",".join(exclude_nodes)])
        cmd.append(str(script_path))
        print("[SUBMIT]", " ".join(map(shlex.quote, cmd)))

        if args.dry_run:
            job_id = str(dry_run_job_id_base + submitted + 1)
            print(f"         [DRY-RUN] fake_job_id={job_id}")
        else:
            try:
                out = run(cmd, debug=args.debug)
                job_id = parse_sbatch_job_id(out)
                print(f"         job_id={job_id} raw={out}")
            except Exception as e:
                if "Invalid node name specified" in str(e):
                    print(
                        f"[WARN] Invalid node '{node_submit}', retry submit without node filter"
                    )
                    retry_cmd = [
                        "sbatch",
                        "--parsable",
                        "--partition",
                        partition,
                        "--export",
                        f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_idx}",
                        "--array",
                        arr,
                    ]
                    if log_output is not None and log_error is not None:
                        retry_cmd.extend(["--output", log_output, "--error", log_error])
                    if dependency_arg is not None:
                        retry_cmd.append(f"--dependency={dependency_arg}")
                    retry_cmd.append(str(script_path))
                    print("[SUBMIT-RETRY]", " ".join(map(shlex.quote, retry_cmd)))
                    out = run(retry_cmd, debug=args.debug)
                    job_id = parse_sbatch_job_id(out)
                    print(f"         job_id={job_id} raw={out}")
                else:
                    raise

        prev_job_id_by_bucket[bucket_idx] = job_id
        submitted += 1

    print(
        f"[DONE] submissions={submitted}, tasks={len(task_ids)}, "
        f"slots={len(slots)}, mode=individual"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
