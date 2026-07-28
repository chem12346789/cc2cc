#!/usr/bin/env python3
import argparse, concurrent.futures as cf, json, re, shlex, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_SCRIPT_DIR = ROOT / "test_script"
TMP_DIR = TEST_SCRIPT_DIR / "tmp"
USABLE_STATES = {"idle", "mix", "mixed"}


def run(command, timeout_sec=None, debug=False):
    if debug:
        print("[DEBUG] RUN:", " ".join(map(shlex.quote, command)))
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=timeout_sec
    )
    if debug and result.stdout.strip():
        print(f"[DEBUG] STDOUT:\n{result.stdout.strip()}")
    if debug and result.stderr.strip():
        print(f"[DEBUG] STDERR:\n{result.stderr.strip()}")
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(shlex.quote, command))}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


def norm_node(node_text):
    return (
        node_text.strip()
        .rstrip(",")
        .replace("*", "")
        .replace("~", "")
        .replace("+", "")
        .split()[0]
    )


def safe_name(name_text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name_text.strip()).strip(".-") or "run"


def resolve_path(path_str):
    candidate_path = Path(path_str).expanduser()
    if candidate_path.is_absolute():
        return candidate_path
    for base_dir in (TEST_SCRIPT_DIR, ROOT, Path.cwd()):
        resolved_path = (base_dir / candidate_path).resolve()
        if resolved_path.exists():
            return resolved_path
    return (ROOT / candidate_path).resolve()


def sbatch_field(text, field_keys):
    for raw_line in text.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line.startswith("#SBATCH"):
            continue
        directive_body = stripped_line[len("#SBATCH") :].strip()
        for directive_key in field_keys:
            if directive_body.startswith(directive_key + "="):
                return directive_body.split("=", 1)[1].strip()
            if directive_body.startswith(directive_key + " "):
                return directive_body.split(None, 1)[1].strip()


def array_ids(array_spec):
    base_spec = array_spec.strip().split("%", 1)[0]
    if base_spec.startswith("[") and base_spec.endswith("]"):
        base_spec = base_spec[1:-1]
    task_id_set = set()
    for token in filter(None, (segment.strip() for segment in base_spec.split(","))):
        range_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)(?::(\d+))?", token)
        if range_match:
            start_id = int(range_match[1])
            end_id = int(range_match[2])
            step = int(range_match[3] or 1)
            if step <= 0:
                raise ValueError(f"Invalid array step in {token!r}")
            task_id_set.update(
                range(
                    start_id,
                    end_id + (1 if start_id <= end_id else -1),
                    step if start_id <= end_id else -step,
                )
            )
        elif re.fullmatch(r"-?\d+", token):
            task_id_set.add(int(token))
        else:
            raise ValueError(f"Unsupported array token: {token}")
    if not task_id_set:
        raise ValueError(f"No task IDs parsed from array spec: {base_spec}")
    return sorted(task_id_set)


def array_limit(array_spec):
    limit_match = re.search(r"%\s*(\d+)\s*$", array_spec.strip())
    return int(limit_match[1]) if limit_match and int(limit_match[1]) > 0 else None


def hostnames(expr, debug=False):
    try:
        rows = run(["scontrol", "show", "hostnames", expr], debug=debug).splitlines()
        return [norm_node(x) for x in rows if x.strip()]
    except Exception:
        return [norm_node(x) for x in expr.split(",") if norm_node(x)]


def partition_nodes(partition, states=None, debug=False):
    rows = run(
        ["sinfo", "-h", "-N", "-p", partition, "-o", "%n|%t"], debug=debug
    ).splitlines()
    nodes = []
    for row in filter(None, (x.strip() for x in rows)):
        node, _, state = row.partition("|")
        st = state.strip().lower().rstrip("*~+#")
        if states is None or st in states:
            nodes += hostnames(norm_node(node), debug)
    out = list(dict.fromkeys(nodes))
    if not out:
        raise RuntimeError(f"No nodes found in partition {partition!r}")
    return out


def load_model_filter(text, key):
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("#")
            or "load_model_args" not in line
            or "--load" not in line
        ):
            continue
        try:
            toks = shlex.split(line, comments=True)
        except ValueError:
            toks = line.split()
        value_candidates = [
            token.split("=", 1)[1]
            for token in toks
            if token.startswith("load_model_args=")
        ] or [line]
        for candidate_value in value_candidates:
            try:
                value_tokens = shlex.split(candidate_value, comments=True)
            except ValueError:
                value_tokens = candidate_value.split()
            for index, token in enumerate(value_tokens):
                if token == f"--{key}" and index + 1 < len(value_tokens):
                    return value_tokens[index + 1]
                if token.startswith(f"--{key}="):
                    return token.split("=", 1)[1]


def molecule_names(text):
    molecule_list_match = re.search(r"name_mol_input_list\s*=\s*\((.*?)\)", text, re.S)
    if not molecule_list_match:
        return []
    try:
        molecule_values = shlex.split(molecule_list_match.group(1), comments=True)
    except ValueError:
        molecule_values = molecule_list_match.group(1).split()
    molecule_names_list = []
    for molecule_value in molecule_values:
        normalized_name = molecule_value.strip().strip("\"'")
        molecule_names_list.append(
            safe_name(
                normalized_name[len("molecule_") :]
                if normalized_name.startswith("molecule_")
                else normalized_name
            )
        )
    return molecule_names_list


def split_by_time(task_ids, estimated_runtime_by_task, bucket_count):
    missing = [
        task_id for task_id in task_ids if task_id not in estimated_runtime_by_task
    ]
    if missing:
        raise ValueError(
            f"Missing runtime estimate(s): {','.join(map(str, missing[:20]))}"
        )
    task_buckets, bucket_loads = [[] for _ in range(bucket_count)], [0.0] * bucket_count
    for task_id in sorted(
        task_ids,
        key=lambda current_id: (estimated_runtime_by_task[current_id], current_id),
        reverse=True,
    ):
        target_bucket_index = min(
            range(bucket_count),
            key=lambda idx: (bucket_loads[idx], len(task_buckets[idx]), idx),
        )
        task_buckets[target_bucket_index].append(task_id)
        bucket_loads[target_bucket_index] += estimated_runtime_by_task[task_id]
    original_order_index = {task_id: index for index, task_id in enumerate(task_ids)}
    for bucket in task_buckets:
        bucket.sort(
            key=lambda task_id: (
                estimated_runtime_by_task[task_id],
                original_order_index[task_id],
            )
        )
    return task_buckets, bucket_loads


def clean_tmp(debug=False):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for temp_file in TMP_DIR.glob("gpu_probe_*"):
        try:
            temp_file.unlink()
            if debug:
                print(f"[DEBUG] removed old tmp: {temp_file}")
        except Exception:
            pass


def wait_probe_done(job_id, timeout_sec):
    deadline = time.monotonic() + max(1, timeout_sec)
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return True
        time.sleep(1)
    return False


def probe_node(
    node,
    part,
    min_free_gb,
    max_power,
    timeout,
    debug,
    nodes,
    ntasks_per_node,
    cpus_per_task,
):
    tag = uuid.uuid4().hex[:10]
    out_file, err_file = (
        TMP_DIR / f"gpu_probe_{node}_{tag}.out",
        TMP_DIR / f"gpu_probe_{node}_{tag}.err",
    )
    cmd = [
        "sbatch",
        "--parsable",
        "--partition",
        part,
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
    cmd += (
        ["--ntasks-per-node", ntasks_per_node] if ntasks_per_node else ["--ntasks", "1"]
    )
    if cpus_per_task:
        cmd += ["--cpus-per-task", cpus_per_task]
    probe_job_id = None
    try:
        probe_job_id = parse_job_id(run(cmd, debug=debug))
        finished_in_time = wait_probe_done(probe_job_id, timeout)
        text = (
            out_file.read_text().strip()
            if finished_in_time and out_file.exists()
            else ""
        )
    except Exception:
        text = ""
    finally:
        if probe_job_id is not None:
            try:
                run(["scancel", str(probe_job_id)], debug=debug)
            except Exception:
                pass
        for temp_file in (out_file, err_file):
            try:
                temp_file.unlink()
            except Exception:
                pass
    min_memory_mib, usable_gpu_indices = int(min_free_gb * 1024), []
    for line in text.splitlines():
        try:
            free_s, power_s, idx_s = [x.strip() for x in line.split(",")[:3]]
            if int(free_s.split()[0]) >= min_memory_mib and (
                max_power is None or float(power_s.split()[0]) <= max_power
            ):
                usable_gpu_indices.append(int(idx_s.split()[0]))
        except Exception:
            pass
    return node, sorted(set(usable_gpu_indices))


def parse_job_id(s):
    job_id_match = re.search(r"\b(\d+)\b", s)
    if not job_id_match:
        raise RuntimeError(f"Could not parse sbatch job id from: {s!r}")
    return job_id_match[1]


def log_paths(args, script_path, text):
    if args.no_log_override:
        return None, None
    load = safe_name(load_model_filter(text, "load") or "no-load")
    epoch = safe_name(load_model_filter(text, "load_epoch") or "no-epoch")
    log_dir = (Path("log") / safe_name(script_path.stem) / load / epoch).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    base = str(log_dir / "%x-a%a")
    print(f"[INFO] Log path={base}")
    return f"{base}.out", f"{base}.err"


def sbatch_cmd(
    partition,
    task_id,
    script_path,
    dependency,
    log_paths_pair,
    excluded_nodes,
    job_name=None,
    gpu_index=None,
    cpus_per_task=None,
):
    command = [
        "sbatch",
        "--parsable",
        "--partition",
        partition,
        "--array",
        str(task_id),
    ]
    if cpus_per_task:
        command += ["--cpus-per-task", str(cpus_per_task)]
    if gpu_index is not None:
        command += ["--export", f"ALL,FORCE_CUDA_VISIBLE_DEVICES={gpu_index}"]
    if job_name:
        command += ["--job-name", job_name]
    output_log_path, error_log_path = log_paths_pair
    if output_log_path and error_log_path:
        command += ["--output", output_log_path, "--error", error_log_path]
    if dependency:
        command.append(f"--dependency={dependency}")
    if excluded_nodes:
        command += ["--exclude", ",".join(excluded_nodes)]
    return command + [str(script_path)]


def build_parser():
    ap = argparse.ArgumentParser()
    add = ap.add_argument
    add("--script", default="test_direct.bash")
    add("--time-array", required=True)
    add("--array-concurrency", type=int, default=1)
    add("--min-free-memory", type=float, default=15.0)
    add("--max-gpu-power", type=float, default=None)
    add("--probe-timeout-sec", type=int, default=10)
    add("--probe-workers", type=int, default=16)
    add("--max-nodes", type=int, default=0)
    add("--no-log-override", action="store_true")
    add("--debug", action="store_true")
    add("--cpu-only", action="store_true")
    return ap


def choose_slots(
    args,
    part,
    probe_nodes,
    script_nodes,
    script_ntasks_per_node,
    script_cpus_per_task,
    limit,
):
    if args.cpu_only:
        out = []
        for n in probe_nodes:
            hs = hostnames(n, args.debug)
            if hs:
                out.append((hs[0], None))
            if len(out) >= limit:
                break
        return out
    with cf.ThreadPoolExecutor(max_workers=max(1, args.probe_workers)) as ex:
        futs = [
            ex.submit(
                probe_node,
                n,
                part,
                args.min_free_memory,
                args.max_gpu_power,
                args.probe_timeout_sec,
                args.debug,
                script_nodes,
                script_ntasks_per_node,
                script_cpus_per_task,
            )
            for n in probe_nodes
        ]
        out = []
        for n, gpus in (f.result() for f in cf.as_completed(futs)):
            hs = hostnames(n, args.debug)
            if not hs:
                continue
            out.extend((hs[0], g) for g in gpus)
            if len(out) >= limit:
                return out[:limit]
    return out[:limit]


def submit_one(args, part, tid, script, dep, logs, exclude, name, gpu, cpus):
    try:
        return parse_job_id(
            run(
                sbatch_cmd(part, tid, script, dep, logs, exclude, name, gpu, cpus),
                debug=args.debug,
            )
        )
    except Exception as e:
        if "Invalid node name specified" not in str(e):
            raise
        return parse_job_id(
            run(
                sbatch_cmd(part, tid, script, dep, logs, [], name, gpu, cpus),
                debug=args.debug,
            )
        )


def main():
    args = build_parser().parse_args()
    script = resolve_path(args.script)
    if not script.exists():
        raise FileNotFoundError(script)
    text = script.read_text()

    clean_tmp(args.debug)
    log_path_pair = log_paths(args, script, text)
    partition, array_spec = sbatch_field(text, ["-p", "--partition"]), sbatch_field(
        text, ["-a", "--array"]
    )
    if not partition or not array_spec:
        raise RuntimeError("Partition/array not found in script")

    task_ids = array_ids(array_spec)
    slot_limit = max(1, array_limit(array_spec) or args.array_concurrency)
    excluded_nodes = set(
        hostnames(sbatch_field(text, ["-x", "--exclude"]) or "", args.debug)
    )
    load_name, mol_names = safe_name(
        load_model_filter(text, "load") or "no-load"
    ), molecule_names(text)
    script_nodes = sbatch_field(text, ["-N", "--nodes"])
    script_ntasks_per_node = sbatch_field(text, ["--ntasks-per-node"])
    script_cpus_per_task = sbatch_field(text, ["-c", "--cpus-per-task"])

    probe_nodes, submission_pool = partition_nodes(
        partition, USABLE_STATES, args.debug
    ), partition_nodes(partition, None, args.debug)
    if excluded_nodes:
        probe_nodes = [
            node_name for node_name in probe_nodes if node_name not in excluded_nodes
        ]
        submission_pool = list(dict.fromkeys([*submission_pool, *excluded_nodes]))
    if args.max_nodes > 0:
        probe_nodes = probe_nodes[: args.max_nodes]

    slots = choose_slots(
        args,
        partition,
        probe_nodes,
        script_nodes,
        script_ntasks_per_node,
        script_cpus_per_task,
        slot_limit,
    )
    if not slots:
        print("[WARN] No usable slots found. Nothing submitted.")
        return 0

    with resolve_path(args.time_array).open() as f:
        cfg = json.load(f)
    estimated_runtime_by_task, cpus_per_task_by_task = {
        task_id: float(runtime) for task_id, runtime in zip(task_ids, cfg["time_array"])
    }, cfg["cpus_per_task"]
    task_buckets, bucket_loads = split_by_time(
        task_ids, estimated_runtime_by_task, len(slots)
    )
    assignment_by_task = {
        task_id: (
            bucket_index + 1,
            slots[bucket_index][0],
            slots[bucket_index][1],
            bucket_loads[bucket_index],
        )
        for bucket_index, bucket_tasks in enumerate(task_buckets)
        for task_id in bucket_tasks
    }

    previous_job_by_bucket, submitted_jobs = {}, []
    for task_id in task_ids:
        bucket_index, node_name, gpu_index, bucket_load = assignment_by_task[task_id]
        dependency = (
            f"afterany:{previous_job_by_bucket[bucket_index]}"
            if bucket_index in previous_job_by_bucket
            else None
        )
        molecule_name = (
            mol_names[task_id] if 0 <= task_id < len(mol_names) else f"task-{task_id}"
        )
        exclude = (
            []
            if args.cpu_only
            else [node for node in submission_pool if node != node_name]
        )
        submitted_job_id = submit_one(
            args,
            partition,
            task_id,
            script,
            dependency,
            log_path_pair,
            exclude,
            safe_name(f"{load_name}-{molecule_name}"),
            gpu_index,
            cpus_per_task_by_task[task_id],
        )
        previous_job_by_bucket[bucket_index] = submitted_job_id
        submitted_jobs.append((task_id, bucket_index, submitted_job_id))

    job_ids_by_bucket, task_ids_by_bucket = {}, {}
    for task_id, bucket_index, job_id in submitted_jobs:
        job_ids_by_bucket.setdefault(bucket_index, []).append(job_id)
        task_ids_by_bucket.setdefault(bucket_index, []).append(str(task_id))
    print(
        "[PID-SUMMARY]",
        " ".join(" ".join(v) for _, v in sorted(job_ids_by_bucket.items())),
    )
    print(
        "[SLURM_ARRAY_TASK_ID]",
        " ".join(f"{k}:{','.join(v)}" for k, v in sorted(task_ids_by_bucket.items())),
    )
    print(
        f"[DONE] submissions={len(submitted_jobs)}, tasks={len(task_ids)}, slots={len(slots)}, mode=individual"
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
