import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import scripts.direct_run_per_gpu as direct


class DirectRunTests(unittest.TestCase):
    def test_force_profile_selects_molecule_and_redirects_logs(self):
        helper = direct.submit.ROOT / "scripts" / "__lib__" / "test_job.sh"
        script = direct.submit.ROOT / "scripts" / "test" / "test_force.bash"
        self.assertIn("select_molecule_profile force", script.read_text())
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.update(
                REPO_ROOT=directory, SLURM_ARRAY_TASK_ID="0",
                load_model_args="--load atom-7297 --load_epoch 1115",
            )
            result = subprocess.run(
                ["bash", "-e", "-c",
                 'source "$1"; select_molecule_profile force; '
                 'printf "molecule=%s\\n" "$name_mol_input"; echo profile-error >&2',
                 "profile-test", str(helper)],
                env=env, text=True, capture_output=True, check=True, timeout=5,
            )
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")
            logs = Path(directory) / "log" / "atom-7297" / "1115"
            self.assertEqual(
                (logs / "Force_0.out").read_text(),
                "array_index=0\nmolecule=molecule_Force\n",
            )
            self.assertEqual((logs / "Force_0.err").read_text(), "profile-error\n")

    def test_force_script_plan(self):
        with patch("builtins.print") as output:
            self.assertEqual(direct.main([
                "--script", "test/test_force.bash", "--gpus", "0", "--dry-run",
            ]), 0)
        lines = [call.args[0] for call in output.call_args_list]
        self.assertIn("jobs=1", lines[0])
        self.assertIn("concurrency=1/GPU total-concurrency=1 cpus/job=20", lines[0])
        self.assertEqual(lines[1], "[DRY-RUN] MODEL_INDEX=0 SLURM_ARRAY_TASK_ID=0")

    def test_background_launch_logging_and_shutdown(self):
        for job_status in (0, 7, None):
            with self.subTest(job_status=job_status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                smi = root / "nvidia-smi"
                smi.write_text("#!/bin/sh\nprintf '0, GPU-a, 24000, 10\\n'\n")
                smi.chmod(0o755)
                script = root / "job.bash"
                script.write_text(
                    "#SBATCH -a [0]%1\n"
                    "echo job-output\n"
                    "echo job-error >&2\n"
                    + ("sleep 10\n" if job_status is None else f"exit {job_status}\n")
                )
                logs = root / "logs"
                env = os.environ.copy()
                env.update(PATH=f"{root}:{env['PATH']}", PYSCF_TMPDIR=str(root / "scratch"))
                launcher_pid = None
                try:
                    parent = subprocess.run(
                        [sys.executable, str(Path(direct.__file__).resolve()),
                         "--script", str(script), "--gpus", "0",
                         "--log-dir", str(logs), "--poll-interval", "0.01"],
                        env=env, text=True, capture_output=True, timeout=5,
                    )
                    self.assertEqual(parent.returncode, 0, parent.stdout + parent.stderr)
                    self.assertIn("[BACKGROUND]", parent.stdout)
                    launcher_pid = int((logs / "launcher.pid").read_text())
                    deadline = time.monotonic() + 5
                    while not (logs / "model-0-task-0.out").exists():
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
                    if job_status is None:
                        os.kill(launcher_pid, 0)
                        os.kill(launcher_pid, signal.SIGTERM)
                    while not (logs / "exit-code").exists():
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
                    expected = 130 if job_status is None else int(job_status != 0)
                    self.assertEqual(int((logs / "exit-code").read_text()), expected)
                    text = (logs / "launcher.log").read_text()
                    self.assertIn("[PLAN]", text)
                    self.assertIn("[START]", text)
                    self.assertIn("[STOP]" if job_status is None else "[SUMMARY]", text)
                    self.assertIn("job-output", (logs / "model-0-task-0.out").read_text())
                    self.assertIn("job-error", (logs / "model-0-task-0.err").read_text())
                finally:
                    if launcher_pid is not None and not (logs / "exit-code").exists():
                        try:
                            os.kill(launcher_pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass

    def test_gpu_filters_and_uuid_selection(self):
        result = subprocess.CompletedProcess([], 0, stdout=(
            "0, GPU-a, 20000, 20\n"
            "1, GPU-b, 10000, 10\n"
            "2, GPU-c, 22000, 90\n"
        ))
        with patch.object(direct.subprocess, "run", return_value=result) as query:
            self.assertEqual(direct.query_gpus(None, 15, 50), {"0": "GPU-a"})
            self.assertEqual(direct.query_gpus(["GPU-c"], 15, None), {"2": "GPU-c"})
            with self.assertRaisesRegex(ValueError, "Unknown GPU IDs"):
                direct.query_gpus(["8"], 15, None)
        self.assertNotIn("sbatch", query.call_args.args[0])
        self.assertEqual(query.call_args.kwargs["timeout"], 15)

    def test_missing_power_requires_explicit_error_only_when_filtered(self):
        result = subprocess.CompletedProcess([], 0, stdout="0, GPU-a, 20000, [N/A]\n")
        with patch.object(direct.subprocess, "run", return_value=result):
            self.assertEqual(direct.query_gpus(None, 15, None), {"0": "GPU-a"})
            with self.assertRaisesRegex(ValueError, "cannot report power"):
                direct.query_gpus(None, 15, 50)

    def test_dry_run_real_diet100_script_without_gpu_or_file_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory) / "logs"
            with (
                patch.object(direct, "query_gpus") as query,
                patch.object(direct.subprocess, "Popen") as launch,
                patch("builtins.print") as output,
            ):
                status = direct.main([
                    "--script", "test/test_diet100.bash", "--gpus", "0,1",
                    "--dry-run", "--log-dir", str(logs),
                ])
            self.assertEqual(status, 0)
            self.assertFalse(logs.exists())
            query.assert_not_called()
            launch.assert_not_called()
            lines = [str(call.args[0]) for call in output.call_args_list]
            self.assertIn("jobs=220", lines[0])
            self.assertIn("concurrency=3/GPU total-concurrency=6 cpus/job=8", lines[0])
            self.assertEqual(len([line for line in lines if "[DRY-RUN]" in line]), 220)
            self.assertIn("SLURM_ARRAY_TASK_ID=54", lines[-1])

    def test_timing_order_zero_skip_models_and_concurrency_override(self):
        with tempfile.TemporaryDirectory() as directory:
            timings = Path(directory) / "times.json"
            timings.write_text(json.dumps({"time_array": [2, 0, 8]}))
            with patch("builtins.print") as output:
                direct.main([
                    "--array", "0-2%1", "--array-concurrency", "2",
                    "--gpus", "0,1", "--model-indices", "0,2",
                    "--time-array", str(timings), "--dry-run",
                ])
            lines = [str(call.args[0]) for call in output.call_args_list]
            self.assertIn("concurrency=2/GPU total-concurrency=4", lines[0])
            self.assertEqual(lines[1:], [
                "[DRY-RUN] MODEL_INDEX=0 SLURM_ARRAY_TASK_ID=2",
                "[DRY-RUN] MODEL_INDEX=0 SLURM_ARRAY_TASK_ID=0",
                "[DRY-RUN] MODEL_INDEX=2 SLURM_ARRAY_TASK_ID=2",
                "[DRY-RUN] MODEL_INDEX=2 SLURM_ARRAY_TASK_ID=0",
            ])

    def test_respects_inherited_gpu_mask(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-b"}),
            patch.object(direct, "query_gpus") as query,
            patch("builtins.print") as output,
        ):
            self.assertEqual(direct.main(["--dry-run", "--array", "0"]), 0)
        query.assert_not_called()
        self.assertIn("GPUs=GPU-b", output.call_args_list[0].args[0])

    def test_empty_gpu_mask_and_invalid_options_fail(self):
        for options in (
            ["--gpus", ""], ["--cpus-per-task", "0"],
            ["--array-concurrency", "0"], ["--poll-interval", "nan"],
            ["--array=-1"], ["--model-indices", "0,0"],
        ):
            with self.subTest(options=options), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    direct.main(["--dry-run", *options])
                self.assertEqual(error.exception.code, 2)

    def test_no_usable_gpu_fails_without_launching(self):
        with (
            patch.object(direct, "query_gpus", return_value={}),
            patch.object(direct.subprocess, "Popen") as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "No local GPUs"):
                direct.main(["--gpus", "0"])
            launch.assert_not_called()

    def test_local_bash_jobs_environment_logs_and_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "job.bash"
            script.write_text(
                "#SBATCH -a [0-2]%2\n"
                "#SBATCH --cpus-per-task=3\n"
                'printf "%s|%s|%s|%s|%s|%s\\n" "$MODEL_INDEX" '
                '"$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK" '
                '"$CUDA_VISIBLE_DEVICES" "$FORCE_CUDA_VISIBLE_DEVICES" "$PWD"\n'
                'test -d "$PYSCF_TMPDIR" || exit 9\n'
                'printf "%s\\n" "$PYSCF_TMPDIR" >&2\n'
                'if [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then exit 7; fi\n'
            )
            logs = root / "logs"
            with (
                patch.object(direct, "query_gpus", return_value={"0": "GPU-a", "1": "GPU-b"}),
                patch.dict(os.environ, {"PYSCF_TMPDIR": str(root / "scratch")}),
                patch("builtins.print"),
            ):
                status = direct.main([
                    "--script", str(script), "--gpus", "0,1", "--log-dir", str(logs),
                    "--poll-interval", "0.01", "--model-indices", "0,1",
                    "--foreground",
                ])
            self.assertEqual(status, 1)
            self.assertEqual(len(list(logs.glob("*.out"))), 6)
            scratch_paths = set()
            for model in range(2):
                for task in range(3):
                    stem = f"model-{model}-task-{task}"
                    fields = (logs / f"{stem}.out").read_text().strip().split("|")
                    self.assertEqual(fields[:3], [str(model), str(task), "3"])
                    self.assertIn(fields[3], ("GPU-a", "GPU-b"))
                    self.assertEqual(fields[3], fields[4])
                    self.assertEqual(fields[5], str(direct.submit.ROOT))
                    scratch = (logs / f"{stem}.err").read_text().strip()
                    self.assertTrue(Path(scratch).is_dir())
                    scratch_paths.add(scratch)
            self.assertEqual(len(scratch_paths), 6)

    def check_scheduler(self, concurrency_per_gpu):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processes = []
            clock = [0]
            max_active_by_gpu = {}

            def advance(seconds):
                clock[0] += 1

            def launch(*args, **kwargs):
                process = Mock()
                end = clock[0] + 2
                process.poll.side_effect = lambda: 0 if clock[0] >= end else None
                processes.append((process, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                gpu_counts = {}
                for active, gpu in processes:
                    if active.poll() is None:
                        gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
                for gpu, active_count in gpu_counts.items():
                    max_active_by_gpu[gpu] = max(
                        max_active_by_gpu.get(gpu, 0), active_count,
                    )
                self.assertLessEqual(max(gpu_counts.values()), concurrency_per_gpu)
                self.assertLessEqual(
                    sum(proc.poll() is None for proc, _ in processes),
                    concurrency_per_gpu * 2,
                )
                return process

            with (
                patch.object(direct, "query_gpus", return_value={"0": "GPU-a", "1": "GPU-b"}),
                patch.object(direct.subprocess, "Popen", side_effect=launch),
                patch.object(direct.time, "sleep", side_effect=advance),
                patch("builtins.print"),
            ):
                status = direct.run_jobs(
                    root / "job.bash", [(0, task) for task in range(5)],
                    {"0": "GPU-a", "1": "GPU-b"}, concurrency_per_gpu, 1, root,
                    root / "scratch", 15, None, 0.01,
                )
            self.assertEqual(status, 0)
            self.assertEqual(len(processes), 5)
            self.assertEqual(max(max_active_by_gpu.values()), concurrency_per_gpu)

    def test_scheduler_respects_concurrency_per_gpu(self):
        for concurrency_per_gpu in (1, 2):
            with self.subTest(concurrency_per_gpu=concurrency_per_gpu):
                self.check_scheduler(concurrency_per_gpu)

    def test_scheduler_waits_for_gpu_to_meet_filters_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.poll.return_value = 0
            with (
                patch.object(direct, "query_gpus", side_effect=[{}, {"0": "GPU-a"}]) as query,
                patch.object(direct.subprocess, "Popen", return_value=process) as launch,
                patch.object(direct.time, "sleep") as sleep,
                patch("builtins.print"),
            ):
                status = direct.run_jobs(
                    root / "job.bash", [(0, 0)], {"0": "GPU-a"}, 1, 1,
                    root, root / "scratch", 15, 50, 0.01,
                )
            self.assertEqual(status, 0)
            self.assertEqual(query.call_count, 2)
            query.assert_called_with(["GPU-a"], 15, 50)
            launch.assert_called_once()
            sleep.assert_any_call(0.01)

    def test_cancellation_terminates_only_launched_process_group(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("bash", 5), 0]
        with patch.object(direct.os, "killpg") as kill:
            direct.stop_jobs({process: ("0", 0, 0)})
        self.assertEqual(
            [call.args for call in kill.call_args_list],
            [(12345, signal.SIGTERM), (12345, signal.SIGKILL)],
        )


if __name__ == "__main__":
    unittest.main()
