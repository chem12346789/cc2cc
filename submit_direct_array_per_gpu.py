#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parent
TEST_SCRIPT_DIR = ROOT / "test_script"
TMP_DIR = TEST_SCRIPT_DIR / "tmp"
USABLE_STATES = {"idle", "mix", "mixed"}


def run(cmd: list[str], timeout_sec: int | None = None, debug: bool = False) -> str:
    if debug:
        print("[DEBUG] RUN:", " ".join(map(shlex.quote, cmd)))
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
    if debug and p.stdout.strip():
        print(f"[DEBUG] STDOUT:\n{p.stdout.strip()}")
    if debug and p.stderr.strip():
        print(f"[DEBUG] STDERR:\n{p.stderr.strip()}")
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(shlex.quote, cmd))}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def norm_node(token: str) -> str:
    return token.strip().rstrip(",").replace("*", "").replace("~", "").replace("+", "").split()[0]


def preview(items, n: int = 20) -> str:
    vals = [str(x) for x in items]
    return ",".join(vals[:n]) + (f",...(+{len(vals)-n})" if len(vals) > n else "")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip(".-") or "run"


def resolve_path(s: str) -> Path:
    p = Path(s).expanduser()
    if p.is_absolute():
        return p
    for b in (TEST_SCRIPT_DIR, ROOT, Path.cwd()):
        q = (b / p).resolve()
        if q.exists():
            return q
    return (ROOT / p).resolve()


def sbatch_field(text: str, keys: list[str]) -> str | None:
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


def fmt_ids(ids: list[int], keep_order: bool = False) -> str:
    vals = list(dict.fromkeys(ids)) if keep_order else sorted(set(ids))
    if not vals:
        raise ValueError("Empty task list")
    out: list[str] = []
    a = b = vals[0]
    for x in vals[1:]:
        if x == b + 1:
            b = x
        else:
            out.append(str(a) if a == b else f"{a}-{b}")
            a = b = x
    out.append(str(a) if a == b else f"{a}-{b}")
    return ",".join(out)


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
        nodes += hostnames(norm_node(node), debug)
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
        vals = [t.split("=", 1)[1] for t in toks if t.startswith("load_model_args=")] or [line]
        for v in vals:
            try:
                vtoks = shlex.split(v, comments=True)
            except ValueError:
                vtoks = v.split()
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
    out = []
    for v in vals:
        n = v.strip().strip("\"'")
        out.append(safe_name(n[len("molecule_") :] if n.startswith("molecule_") else n))
    return out


def load_times(path: Path, offset: int = 0) -> dict[int, float]:
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        nums, bad = [], False
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
        return dict(check(int(t), rt) for t, rt in rows)
    return dict(check(offset + i, v) for i, v in enumerate(x for row in rows for x in row))


def split_round_robin(ids: list[int], n: int):
    b = [[] for _ in range(n)]
    for i, tid in enumerate(ids):
        b[i % n].append(tid)
    return b, [float(len(x)) for x in b]


def split_by_time(ids: list[int], est: dict[int, float], n: int):
    missing = [tid for tid in ids if tid not in est]
    if missing:
        raise ValueError(f"Missing runtime estimate(s): {preview(missing)}")
    b, load = [[] for _ in range(n)], [0.0] * n
    for tid in sorted(ids, key=lambda x: (est[x], x), reverse=True):
        i = min(range(n), key=lambda j: (load[j], len(b[j]), j))
        b[i].append(tid)
        load[i] += est[tid]
    order = {tid: i for i, tid in enumerate(ids)}
    for x in b:
        x.sort(key=lambda tid: (est[tid], order[tid]))
    return b, load


def clean_tmp(debug: bool = False) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in TMP_DIR.glob("gpu_probe_*"):
        try:
            p.unlink()
            if debug:
                print(f"[DEBUG] removed old tmp: {p}")
        except Exception:
            pass


def probe_node(node: str, partition: str, min_free_gb: float, max_power: float | None, timeout: int, debug: bool):
    tag = uuid.uuid4().hex[:10]
    out_file = TMP_DIR / f"gpu_probe_{node}_{tag}.out"
    err_file = TMP_DIR / f"gpu_probe_{node}_{tag}.err"
    cmd = [
        "timeout", f"{timeout}s", "sbatch", "--parsable", "--wait", "--partition", partition, "--nodelist", node,
        "--nodes", "1", "--ntasks", "1", "--time", "00:00:30", "--job-name", "probe-gpu-mem", "--output",
        str(out_file), "--error", str(err_file), "--wrap", "nvidia-smi --query-gpu=memory.free,power.draw,index --format=csv,noheader,nounits",
    ]
    try:
        run(cmd, timeout_sec=timeout + 3, debug=debug)
        text = out_file.read_text().strip() if out_file.exists() else ""
    except Exception:
        text = ""
    finally:
        for p in (out_file, err_file):
            try:
                p.unlink()
            except Exception:
                pass
    min_mib, gpus = int(min_free_gb * 1024), []
    for line in text.splitlines():
        try:
            free_s, power_s, idx_s = [x.strip() for x in line.split(",")[:3]]
            power_ok = max_power is None or float(power_s.split()[0]) <= max_power
            if int(free_s.split()[0]) >= min_mib and power_ok:
                gpus.append(int(idx_s.split()[0]))
        except Exception:
            pass
    return node, sorted(set(gpus))


def parse_job_id(s: str) -> str:
    m = re.search(r"\b(\d+)\b", s)
    if not m:
        raise RuntimeError(f"Could not parse sbatch job id from: {s!r}")
    return m[1]


def log_paths(args: argparse.Namespace, script_path: Path, script_text: str):
    if args.no_log_override:
        return None, None
    load = safe_name(load_model_name(script_text) or "no-load")
    log_dir = (Path("log") / safe_name(script_path.stem) / load).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    base = str(log_dir / "%x-a%a")
    print(f"[INFO] Log path={base}")
    return f"{base}.out", f"{base}.err"


def sbatch_cmd(partition: str, arr: str, script: Path, dep: str | None, logs, exclude: list[str], job_name: str | None = None, gpu_idx: int | None = None):
    cmd = ["sbatch", "--parsable", "--partition", partition, "--array", arr]
    if gpu_idx is not None:
        cmd += ["--export", f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_idx}"]
    if job_name:
        cmd += ["--job-name", job_name]
    out, err = logs
    if out and err:
        cmd += ["--output", out, "--error", err]
    if dep:
        cmd.append(f"--dependency={dep}")
    if exclude:
        cmd += ["--exclude", ",".join(exclude)]
    return cmd + [str(script)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add = ap.add_argument
    add("--script", default="test_direct.bash")
    add("--partition", default=None)
    add("--array", default=None)
    add("--min-free-memory", type=float, default=15.0)
    add("--max-gpu-power", type=float, default=None)
    add("--array-concurrency", type=int, default=1)
    add("--time-array", default=None)
    add("--time-array-offset", type=int, default=0)
    add("--no-time-balance", action="store_true")
    add("--probe-timeout-sec", type=int, default=60)
    add("--probe-workers", type=int, default=16)
    add("--max-nodes", type=int, default=0)
    add("--log-dir", default="log")
    add("--no-log-override", action="store_true")
    add("--debug", action="store_true")
    add("--dry-run", action="store_true")
    add("--cpu-only", action="store_true")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    script = resolve_path(a.script)
    if not script.exists():
        raise FileNotFoundError(script)
    text = script.read_text()
    clean_tmp(a.debug)
    logs = log_paths(a, script, text)

    part = a.partition or sbatch_field(text, ["-p", "--partition"])
    arr = a.array or sbatch_field(text, ["-a", "--array"])
    if not part or not arr:
        raise RuntimeError("Partition/array not found in script; use --partition/--array")
    ids = array_ids(arr)
    limit = max(1, array_limit(arr) or a.array_concurrency)
    excl = set(hostnames(sbatch_field(text, ["-x", "--exclude"]) or "", a.debug))
    load_name, mol_names = safe_name(load_model_name(text) or "no-load"), molecule_names(text)

    probe_nodes = partition_nodes(part, USABLE_STATES, a.debug)
    submit_pool = partition_nodes(part, None, a.debug)
    if excl:
        probe_nodes = [n for n in probe_nodes if n not in excl]
        submit_pool = list(dict.fromkeys([*submit_pool, *excl]))
    if a.max_nodes > 0:
        probe_nodes = probe_nodes[: a.max_nodes]

    slots: list[tuple[str, int | None]] = []
    if a.cpu_only:
        cpus = list(dict.fromkeys(hostnames(n, a.debug)[0] for n in probe_nodes if hostnames(n, a.debug)))
        slots = [(n, None) for n in cpus[: min(len(cpus), limit)]]
    else:
        with cf.ThreadPoolExecutor(max_workers=max(1, a.probe_workers)) as ex:
            futs = [ex.submit(probe_node, n, part, a.min_free_memory, a.max_gpu_power, a.probe_timeout_sec, a.debug) for n in probe_nodes]
            gpu_slots = [(n, g) for n, gpus in (f.result() for f in cf.as_completed(futs)) for g in gpus]
        valid = [(hostnames(n, a.debug)[0], g) for n, g in gpu_slots if hostnames(n, a.debug)]
        slots = valid[: min(len(valid), limit)]
    if not slots:
        print("[WARN] No usable slots found. Nothing submitted.")
        return 0

    est, time_path = {}, None
    explicit_time = a.time_array is not None
    if not a.no_time_balance:
        if a.time_array is None:
            auto = TEST_SCRIPT_DIR / "task_times_0_54.csv"
            time_path = auto if auto.exists() else None
        elif a.time_array.strip().lower() not in {"", "none", "off", "false", "0"}:
            time_path = resolve_path(a.time_array)
    if time_path:
        if not time_path.exists() and explicit_time:
            raise FileNotFoundError(time_path)
        if time_path.exists():
            est = load_times(time_path, a.time_array_offset)
            miss = [tid for tid in ids if tid not in est]
            if miss and explicit_time:
                raise RuntimeError(f"Runtime estimate file {time_path} missing task ID(s): {preview(miss)}")
            if miss:
                est, time_path = {}, None
    if time_path:
        ids = [tid for tid in ids if est[tid] != 0]
        if not ids:
            print("[WARN] All selected tasks have est_time == 0. Nothing submitted.")
            return 0
        buckets, loads = split_by_time(ids, est, len(slots))
    else:
        buckets, loads = split_round_robin(ids, len(slots))

    assign = {tid: (i + 1, slots[i][0], slots[i][1], loads[i]) for i, b in enumerate(buckets) for tid in b}
    order = {tid: i for i, tid in enumerate(ids)}
    plan = [(tid, *assign[tid]) for tid in ids]
    if time_path:
        plan.sort(key=lambda x: (est[x[0]], order[x[0]]))

    prev, submitted, fake = {}, 0, 900_000_000
    jobs: list[tuple[int, int, str]] = []
    for tid, bi, node, gpu, load in plan:
        dep = f"afterany:{prev[bi]}" if bi in prev else None
        mn = mol_names[tid] if 0 <= tid < len(mol_names) else f"task-{tid}"
        name = safe_name(f"{load_name}-{mn}")
        exclude = [] if a.cpu_only else [n for n in submit_pool if n != node]
        cmd = sbatch_cmd(part, str(tid), script, dep, logs, exclude, job_name=name, gpu_idx=gpu)
        if a.dry_run:
            job_id = str(fake + submitted + 1)
        else:
            try:
                job_id = parse_job_id(run(cmd, debug=a.debug))
            except Exception as e:
                if "Invalid node name specified" not in str(e):
                    raise
                job_id = parse_job_id(run(sbatch_cmd(part, str(tid), script, dep, logs, [], job_name=name, gpu_idx=gpu), debug=a.debug))
        prev[bi] = job_id
        jobs.append((tid, bi, job_id))
        submitted += 1

    if jobs:
        pids, tids = {}, {}
        for tid, bi, pid in jobs:
            pids.setdefault(bi, []).append(pid)
            tids.setdefault(bi, []).append(str(tid))
        print("[PID-SUMMARY]", " ".join(" ".join(v) for _, v in sorted(pids.items())))
        print("[SLURM_ARRAY_TASK_ID]", " ".join(f"{k}:{','.join(v)}" for k, v in sorted(tids.items())))
    print(f"[DONE] submissions={submitted}, tasks={len(ids)}, slots={len(slots)}, mode=individual")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
