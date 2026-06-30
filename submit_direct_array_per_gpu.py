#!/usr/bin/env python3
"""Submit Slurm array tasks on free GPU slots with runtime-balanced scheduling.

- Reads partition/array settings from a target sbatch script.
- Probes GPU memory via tiny sbatch jobs.
- Splits tasks across selected GPU slots, using task runtime estimates when present.
- Submits tasks individually, shortest estimated jobs first, with per-slot dependency chains.
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
USABLE_STATES = {"idle", "mix", "mixed"}


def run(cmd: list[str], timeout_sec: int | None = None, debug: bool = False) -> str:
    if debug:
        print("[DEBUG] RUN:", " ".join(map(shlex.quote, cmd)))
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
    if debug:
        if p.stdout.strip():
            print(f"[DEBUG] STDOUT:\n{p.stdout.strip()}")
        if p.stderr.strip():
            print(f"[DEBUG] STDERR:\n{p.stderr.strip()}")
    if p.returncode:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(map(shlex.quote, cmd))}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def norm_node(token: str) -> str:
    return token.strip().rstrip(",").replace("*", "").replace("~", "").replace("+", "").split()[0]


def preview(items: Iterable[object], n: int = 20) -> str:
    vals = [str(x) for x in items]
    return ",".join(vals[:n]) + (f",...(+{len(vals) - n})" if len(vals) > n else "")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip(".-") or "run"


def resolve_path(s: str) -> Path:
    p = Path(s).expanduser()
    if p.is_absolute():
        return p
    for base in (TEST_SCRIPT_DIR, ROOT, Path.cwd()):
        q = (base / p).resolve()
        if q.exists():
            return q
    return (ROOT / p).resolve()


def sbatch_field(text: str, keys: Iterable[str]) -> str | None:
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("#SBATCH"):
            continue
        body = line[len("#SBATCH") :].strip()
        for k in keys:
            if body.startswith(k + "="):
                return body.split("=", 1)[1].strip()
            if body.startswith(k + " "):
                return body.split(None, 1)[1].strip()
    return None


def array_ids(spec: str) -> list[int]:
    spec = spec.strip().split("%", 1)[0]
    spec = spec[1:-1] if spec.startswith("[") and spec.endswith("]") else spec
    out: set[int] = set()
    for tok in filter(None, (x.strip() for x in spec.split(","))):
        m = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)(?::(\d+))?", tok)
        if m:
            a, b, step = int(m[1]), int(m[2]), int(m[3] or 1)
            if step <= 0:
                raise ValueError(f"Invalid array step in {tok!r}")
            out.update(range(a, b + (1 if a <= b else -1), step if a <= b else -step))
        elif re.fullmatch(r"-?\d+", tok):
            out.add(int(tok))
        else:
            raise ValueError(f"Unsupported array token: {tok}")
    if not out:
        raise ValueError(f"No task IDs parsed from array spec: {spec}")
    return sorted(out)


def array_limit(spec: str) -> int | None:
    m = re.search(r"%\s*(\d+)\s*$", spec.strip())
    return int(m[1]) if m and int(m[1]) > 0 else None


def fmt_ids(ids: Iterable[int], keep_order: bool = False) -> str:
    vals = list(dict.fromkeys(ids)) if keep_order else sorted(set(ids))
    if not vals:
        raise ValueError("Empty task list")
    ranges: list[str] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = x
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def hostnames(expr: str, debug: bool = False) -> list[str]:
    try:
        return [norm_node(x) for x in run(["scontrol", "show", "hostnames", expr], debug=debug).splitlines() if x.strip()]
    except Exception:
        return [norm_node(x) for x in expr.split(",") if norm_node(x)]


def partition_nodes(partition: str, states: set[str] | None, debug: bool = False) -> list[str]:
    out = run(["sinfo", "-h", "-N", "-p", partition, "-o", "%n|%t"], debug=debug)
    nodes: list[str] = []
    for row in filter(None, (x.strip() for x in out.splitlines())):
        node, _, state = row.partition("|")
        state = state.strip().lower().rstrip("*~+#")
        if states is not None and state not in states:
            continue
        nodes.extend(hostnames(norm_node(node), debug=debug))
    nodes = list(dict.fromkeys(nodes))
    if not nodes:
        raise RuntimeError(f"No nodes found in partition {partition!r}")
    return nodes


def load_model_name(script_text: str) -> str | None:
    for raw in script_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "load_model_args" not in line or "--load" not in line:
            continue
        try:
            toks = shlex.split(line, comments=True)
        except ValueError:
            toks = line.split()
        values = [t.split("=", 1)[1] for t in toks if t.startswith("load_model_args=")] or [line]
        for value in values:
            try:
                vtoks = shlex.split(value, comments=True)
            except ValueError:
                vtoks = value.split()
            for i, tok in enumerate(vtoks):
                if tok == "--load" and i + 1 < len(vtoks):
                    return vtoks[i + 1]
                if tok.startswith("--load="):
                    return tok.split("=", 1)[1]
    return None


def molecule_names(script_text: str) -> list[str]:
    m = re.search(r"name_mol_input_list\s*=\s*\((.*?)\)", script_text, re.S)
    if not m:
        return []
    try:
        vals = shlex.split(m.group(1), comments=True)
    except ValueError:
        vals = m.group(1).split()
    out: list[str] = []
    for v in vals:
        name = v.strip().strip("\"'")
        if name.startswith("molecule_"):
            name = name[len("molecule_") :]
        out.append(safe_name(name))
    return out


def load_times(path: Path, offset: int = 0) -> dict[int, float]:
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        nums: list[float] = []
        bad = False
        for tok in filter(None, re.split(r"[\s,]+", line)):
            try:
                nums.append(float(tok))
            except ValueError:
                bad = True
        if bad and nums:
            raise ValueError(f"Bad runtime-estimate line in {path}: {raw!r}")
        if nums:
            rows.append(nums)
    if not rows:
        raise ValueError(f"No numeric runtime estimates found in {path}")

    def check(tid: int, val: float) -> tuple[int, float]:
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"Invalid runtime for task {tid}: {val}")
        return tid, val

    if len(rows) > 1 and all(len(r) == 2 and float(r[0]).is_integer() for r in rows):
        return dict(check(int(tid), runtime) for tid, runtime in rows)
    return dict(check(offset + i, v) for i, v in enumerate(x for row in rows for x in row))


def split_round_robin(ids: list[int], n: int) -> tuple[list[list[int]], list[float]]:
    buckets = [[] for _ in range(n)]
    for i, tid in enumerate(ids):
        buckets[i % n].append(tid)
    return buckets, [float(len(b)) for b in buckets]


def split_by_time(ids: list[int], est: dict[int, float], n: int) -> tuple[list[list[int]], list[float]]:
    missing = [tid for tid in ids if tid not in est]
    if missing:
        raise ValueError(f"Missing runtime estimate(s): {preview(missing)}")
    buckets = [[] for _ in range(n)]
    loads = [0.0] * n
    for tid in sorted(ids, key=lambda x: (est[x], x), reverse=True):
        i = min(range(n), key=lambda j: (loads[j], len(buckets[j]), j))
        buckets[i].append(tid)
        loads[i] += est[tid]
    order = {tid: i for i, tid in enumerate(ids)}
    for b in buckets:
        b.sort(key=lambda tid: (est[tid], order[tid]))
    return buckets, loads


def clean_tmp(debug: bool = False) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in TMP_DIR.glob("gpu_probe_*"):
        try:
            p.unlink()
            if debug:
                print(f"[DEBUG] removed old tmp: {p}")
        except Exception:
            pass


def probe_node(node: str, partition: str, min_free_gb: float, max_power: float | None, timeout: int, debug: bool) -> tuple[str, list[int]]:
    tag = uuid.uuid4().hex[:10]
    out_file = TMP_DIR / f"gpu_probe_{node}_{tag}.out"
    err_file = TMP_DIR / f"gpu_probe_{node}_{tag}.err"
    cmd = [
        "timeout", f"{timeout}s", "sbatch", "--parsable", "--wait", "--partition", partition,
        "--nodelist", node, "--nodes", "1", "--ntasks", "1", "--time", "00:00:30",
        "--job-name", "probe-gpu-mem", "--output", str(out_file), "--error", str(err_file),
        "--wrap", "nvidia-smi --query-gpu=memory.free,power.draw,index --format=csv,noheader,nounits",
    ]
    try:
        run(cmd, timeout_sec=timeout + 3, debug=debug)
        text = out_file.read_text().strip() if out_file.exists() else ""
        if debug and text:
            print(f"[DEBUG] PROBE STDOUT ({node}):\n{text}")
        if debug and err_file.exists() and err_file.read_text().strip():
            print(f"[DEBUG] PROBE STDERR ({node}):\n{err_file.read_text().strip()}")
    except Exception as e:
        if debug:
            print(f"[DEBUG] Probe failed on {node}: {e}")
        text = ""
    finally:
        for p in (out_file, err_file):
            try:
                p.unlink()
            except Exception:
                pass

    min_mib = int(min_free_gb * 1024)
    gpus: list[int] = []
    for line in text.splitlines():
        try:
            free_s, power_s, idx_s = [x.strip() for x in line.split(",")[:3]]
            free_mib = int(free_s.split()[0])
            gpu_idx = int(idx_s.split()[0])
            power_ok = max_power is None or float(power_s.split()[0]) <= max_power
            if free_mib >= min_mib and power_ok:
                gpus.append(gpu_idx)
        except Exception:
            pass
    return node, sorted(set(gpus))


def parse_job_id(s: str) -> str:
    m = re.search(r"\b(\d+)\b", s)
    if not m:
        raise RuntimeError(f"Could not parse sbatch job id from: {s!r}")
    return m[1]


def log_paths(args: argparse.Namespace, script_path: Path, script_text: str) -> tuple[str | None, str | None]:
    if args.no_log_override:
        print("[INFO] Log override disabled; using #SBATCH -o/-e from target script.")
        return None, None
    load = load_model_name(script_text) or "no-load"
    prefix = safe_name(args.log_prefix or f"{script_path.stem}-{load}-{datetime.now():%Y%m%d-%H%M%S}")
    log_dir = Path(args.log_dir.strip() or ".").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir_s = str(log_dir) if args.log_dir.strip().startswith("~") else (args.log_dir.strip() or ".")
    stem = f"{prefix}-%x-a%a"
    base = f"{log_dir_s.rstrip('/')}/{stem}" if log_dir_s != "." else stem
    print(f"[INFO] Log prefix={prefix}")
    if args.debug:
        print(f"[DEBUG] Parsed --load for log prefix: {load}")
        print(f"[DEBUG] sbatch stdout override: {base}.out")
        print(f"[DEBUG] sbatch stderr override: {base}.err")
    return f"{base}.out", f"{base}.err"


def sbatch_cmd(partition: str, gpu_idx: int, arr: str, script: Path, dep: str | None, logs: tuple[str | None, str | None], exclude: list[str], job_name: str | None = None) -> list[str]:
    cmd = [
        "sbatch", "--parsable", "--partition", partition,
        "--export", f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_idx}", "--array", arr,
    ]
    if job_name:
        cmd += ["--job-name", job_name]
    out, err = logs
    if out and err:
        cmd += ["--output", out, "--error", err]
    if dep:
        cmd.append(f"--dependency={dep}")
    if exclude:
        cmd += ["--exclude", ",".join(exclude)]
    cmd.append(str(script))
    return cmd


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add = ap.add_argument
    add("--script", default="test_direct.bash", help="Slurm bash script path; relative paths prefer test_script/")
    add("--partition", default=None, help="Override partition")
    add("--array", default=None, help="Override array spec")
    add("--min-free-memory", type=float, default=15.0, help="GPU free memory threshold in GB")
    add("--max-gpu-power", type=float, default=None, help="Max GPU power draw in W")
    add("--array-concurrency", type=int, default=1)
    add("--time-array", default=None, help="Runtime estimate file; auto-uses test_script/task_times_0_54.csv when present")
    add("--time-array-offset", type=int, default=0)
    add("--no-time-balance", action="store_true", help="Use round-robin splitting")
    add("--probe-timeout-sec", type=int, default=60)
    add("--probe-workers", type=int, default=16)
    add("--max-nodes", type=int, default=0, help="0 means no limit")
    add("--log-dir", default="log")
    add("--log-prefix", default=None, help="Default: <script-stem>-<load-model>-<YYYYMMDD-HHMMSS>")
    add("--no-log-override", action="store_true")
    add("--debug", action="store_true")
    add("--dry-run", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    script_path = resolve_path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    script_text = script_path.read_text()
    logs = log_paths(args, script_path, script_text)
    clean_tmp(args.debug)

    partition = args.partition or sbatch_field(script_text, ["-p", "--partition"])
    array_spec = args.array or sbatch_field(script_text, ["-a", "--array"])
    if not partition or not array_spec:
        raise RuntimeError("Partition/array not found in script; use --partition/--array")
    task_ids = array_ids(array_spec)
    limit = max(1, array_limit(array_spec) or args.array_concurrency)
    exclude_spec = sbatch_field(script_text, ["-x", "--exclude"]) or ""
    load_name = safe_name(load_model_name(script_text) or "no-load")
    mol_names = molecule_names(script_text)

    print(f"[INFO] Script={script_path.name}; partition={partition}; array={array_spec}; tasks={len(task_ids)}; limit={limit}")
    if args.debug:
        print(f"[DEBUG] Task IDs: {fmt_ids(task_ids)}")
        if mol_names:
            print(f"[DEBUG] Parsed molecule job names: {preview(mol_names)}")

    probe_nodes = partition_nodes(partition, USABLE_STATES, args.debug)
    submit_pool = partition_nodes(partition, None, args.debug)
    excluded = set(hostnames(exclude_spec, args.debug)) if exclude_spec else set()
    if excluded:
        probe_nodes = [n for n in probe_nodes if n not in excluded]
        submit_pool = list(dict.fromkeys([*submit_pool, *excluded]))
        print(f"[INFO] Script --exclude nodes added/removed: {len(excluded)}")
    if args.max_nodes > 0:
        probe_nodes = probe_nodes[: args.max_nodes]
    print(f"[INFO] Probe nodes={len(probe_nodes)}; exclude-pinning node pool={len(submit_pool)}")
    if args.debug:
        print(f"[DEBUG] Probe node preview: {preview(probe_nodes)}")
        print(f"[DEBUG] Exclude-pinning node pool preview: {preview(submit_pool)}")

    print(f"[INFO] Probing GPUs: min_free={args.min_free_memory} GB, workers={args.probe_workers}")
    slots: list[tuple[str, int]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.probe_workers)) as ex:
        futs = [ex.submit(probe_node, n, partition, args.min_free_memory, args.max_gpu_power, args.probe_timeout_sec, args.debug) for n in probe_nodes]
        for done, fut in enumerate(cf.as_completed(futs), 1):
            node, gpus = fut.result()
            if gpus:
                if args.debug:
                    print(f"[DEBUG] {node}: eligible GPU(s) = {gpus}")
                slots += [(node, g) for g in gpus]
            if args.debug and (done % 20 == 0 or done == len(futs)):
                print(f"[INFO] Probe progress: {done}/{len(futs)}")
    if not slots:
        print("[WARN] No eligible GPUs found. Nothing submitted.")
        return 0
    print(f"[INFO] Available eligible GPUs={len(slots)}")

    valid: list[tuple[str, int]] = []
    for node, gpu in slots:
        try:
            names = hostnames(node, args.debug)
            if names:
                valid.append((names[0], gpu))
        except Exception:
            pass
    slots = valid[: min(len(valid), limit)]
    if not slots:
        print("[WARN] No valid eligible GPUs found. Nothing submitted.")
        return 0
    print(f"[INFO] Using GPU slots={len(slots)}")
    if args.debug:
        print(f"[DEBUG] Slot list: {preview(f'{n}:gpu{g}' for n, g in slots)}")

    estimates: dict[int, float] = {}
    time_path: Path | None = None
    explicit_time = args.time_array is not None
    if not args.no_time_balance:
        if args.time_array is None:
            auto = TEST_SCRIPT_DIR / "task_times_0_54.csv"
            time_path = auto if auto.exists() else None
        elif args.time_array.strip().lower() not in {"", "none", "off", "false", "0"}:
            time_path = resolve_path(args.time_array)
    if time_path:
        if not time_path.exists():
            if explicit_time:
                raise FileNotFoundError(time_path)
            time_path = None
        else:
            estimates = load_times(time_path, args.time_array_offset)
            missing = [tid for tid in task_ids if tid not in estimates]
            if missing and explicit_time:
                raise RuntimeError(f"Runtime estimate file {time_path} missing task ID(s): {preview(missing)}")
            if missing:
                print("[WARN] Auto runtime estimates incomplete; using round-robin.")
                time_path = None
                estimates = {}
    if time_path:
        skip = [tid for tid in task_ids if estimates[tid] == 0]
        if skip:
            print(f"[INFO] Skipping est_time==0 task ID(s): {fmt_ids(skip)}")
            task_ids = [tid for tid in task_ids if estimates[tid] != 0]
            if not task_ids:
                print("[WARN] All selected tasks have est_time == 0. Nothing submitted.")
                return 0
        vals = [estimates[tid] for tid in task_ids]
        slow = sorted(task_ids, key=lambda tid: (estimates[tid], tid), reverse=True)[:10]
        print(f"[INFO] Time file={time_path}; total={sum(vals):.2f}, min/max={min(vals):.2f}/{max(vals):.2f}, avg={sum(vals)/len(vals):.2f}")
        if args.debug:
            print("[DEBUG] Slowest:", " ".join(f"{tid}:{estimates[tid]:.2f}" for tid in slow))
        buckets, loads = split_by_time(task_ids, estimates, len(slots))
        nonempty = [load for load, b in zip(loads, buckets) if b]
        print(f"[INFO] Balanced load min/max={min(nonempty):.2f}/{max(nonempty):.2f}")
    else:
        buckets, loads = split_round_robin(task_ids, len(slots))
        print("[INFO] Using round-robin task splitting.")

    assignment: dict[int, tuple[int, str, int, float]] = {}
    for bi, ((node, gpu), ids, load) in enumerate(zip(slots, buckets, loads), 1):
        if not ids:
            continue
        label = "est_load" if time_path else "task_count"
        if args.debug:
            print(f"[DEBUG] bucket {bi}: node={node}, gpu={gpu}, tasks={len(ids)}, {label}={load:.2f}, ids={fmt_ids(ids, True)}")
            if time_path:
                print("[DEBUG] bucket", bi, "task:time", " ".join(f"{tid}:{estimates[tid]:.2f}" for tid in ids))
        for tid in ids:
            assignment[tid] = (bi, node, gpu, load)
    missing_assign = [tid for tid in task_ids if tid not in assignment]
    if missing_assign:
        raise RuntimeError(f"Internal missing assignment(s): {preview(missing_assign)}")

    order = {tid: i for i, tid in enumerate(task_ids)}
    plan = [(tid, *assignment[tid]) for tid in task_ids]
    if time_path:
        plan.sort(key=lambda x: (estimates[x[0]], order[x[0]]))
    bucket_summary = " ".join(f"{i}:{len(b)}/{loads[i-1]:.2f}" for i, b in enumerate(buckets, 1) if b)
    print(f"[INFO] Buckets tasks/load: {bucket_summary}")
    print(f"[INFO] Submit order={'small est_time first' if time_path else 'original array order'}")

    submitted = 0
    submitted_jobs: list[tuple[int, int, str, str]] = []
    prev_by_bucket: dict[int, str] = {}
    fake_base = 900_000_000
    for i, (tid, bi, node, gpu, load) in enumerate(plan, 1):
        dep = f"afterany:{prev_by_bucket[bi]}" if bi in prev_by_bucket else None
        metric = f"est_time={estimates[tid]:.2f}, bucket_est_load={load:.2f}" if time_path else f"bucket_task_count={load:.0f}"
        mol_name = mol_names[tid] if 0 <= tid < len(mol_names) else f"task-{tid}"
        job_name = safe_name(f"{load_name}-{mol_name}")
        if args.debug:
            print(f"[DEBUG] Submit {i}/{len(plan)}: task={tid}, job={job_name}, bucket={bi}, node={node}, gpu={gpu}, dep={dep or 'none'}, {metric}")
        exclude = [n for n in submit_pool if n != node]
        cmd = sbatch_cmd(partition, gpu, str(tid), script_path, dep, logs, exclude, job_name=job_name)
        if args.debug:
            print("[SUBMIT]", " ".join(map(shlex.quote, cmd)))
        if args.dry_run:
            job_id = str(fake_base + submitted + 1)
            if args.debug:
                print(f"[JOBID] task={tid} job={job_name} jobid={job_id} dry_run=1")
        else:
            try:
                out = run(cmd, debug=args.debug)
            except Exception as e:
                if "Invalid node name specified" not in str(e):
                    raise
                print(f"[WARN] Invalid node/exclude list, retry task {tid} without node filter")
                retry = sbatch_cmd(partition, gpu, str(tid), script_path, dep, logs, [], job_name=job_name)
                if args.debug:
                    print("[SUBMIT-RETRY]", " ".join(map(shlex.quote, retry)))
                out = run(retry, debug=args.debug)
            job_id = parse_job_id(out)
            if args.debug:
                print(f"[JOBID] task={tid} job={job_name} jobid={job_id}")
                print(f"         job_id={job_id} raw={out}")
        submitted_jobs.append((tid, bi, job_name, job_id))
        prev_by_bucket[bi] = job_id
        submitted += 1
        if not args.debug and (submitted % 10 == 0 or submitted == len(plan)):
            print(f"[INFO] Submitted {submitted}/{len(plan)}")

    if submitted_jobs:
        bucket_pids: dict[int, list[str]] = {}
        bucket_task_ids: dict[int, list[str]] = {}
        for tid, bi, _job, pid in submitted_jobs:
            bucket_pids.setdefault(bi, []).append(pid)
            bucket_task_ids.setdefault(bi, []).append(str(tid))
        pid_summary = " ".join(
            f"{bi}:{','.join(pids)}" for bi, pids in sorted(bucket_pids.items())
        )
        task_id_summary = " ".join(
            f"{bi}:{','.join(ids)}" for bi, ids in sorted(bucket_task_ids.items())
        )
        print(f"[PID-SUMMARY] {pid_summary}")
        print(f"[SLURM_ARRAY_TASK_ID] {task_id_summary}")
    print(f"[DONE] submissions={submitted}, tasks={len(task_ids)}, slots={len(slots)}, mode=individual")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
