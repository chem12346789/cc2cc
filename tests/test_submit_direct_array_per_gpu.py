import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.submit_direct_array_per_gpu as submit


class GPUProbeTests(unittest.TestCase):
    def run_probe(self, text, *, min_free=15.0, max_power=50.0, done=True,
                  output_exists=True, duration=60):
        with tempfile.TemporaryDirectory() as directory:
            def submit_probe(command, *, debug):
                self.command = command
                if output_exists:
                    Path(command[command.index("--output") + 1]).write_text(text)
                return "123"

            with (
                patch.object(submit, "TMP_DIR", Path(directory)),
                patch.object(submit, "run_cmd", side_effect=submit_probe),
                patch.object(submit, "wait_probe_done", return_value=done) as wait,
                patch.object(submit.subprocess, "run") as cancel,
                patch("builtins.print") as log,
            ):
                result = submit.probe_node_gpus(
                    "node1", "gpu", min_free, max_power, duration, False,
                    None, None, None,
                )
                wait.assert_called_once_with("123", 2 * duration + 10)
                cancel.assert_called_once_with(
                    ["scancel", "123"], text=True, capture_output=True, check=False,
                )
                self.assertEqual(list(Path(directory).iterdir()), [])
                return result, log

    def test_transient_power_peak_rejects_gpu(self):
        result, _ = self.run_probe(
            "20000, 20, 0\n20000, 80, 0\n20000, 20, 0\nPROBE_COMPLETE\n"
        )
        self.assertEqual(result, ("node1", []))

    def test_transient_low_free_memory_rejects_gpu(self):
        result, _ = self.run_probe(
            "20000, 20, 0\n10000, 20, 0\n20000, 20, 0\nPROBE_COMPLETE\n"
        )
        self.assertEqual(result, ("node1", []))

    def test_extrema_are_per_gpu_and_thresholds_are_inclusive(self):
        result, _ = self.run_probe(
            "20000, 20, 2\n20000, 20, 0\n20000, 80, 1\n"
            "15360, 50, 2\n20000, 40, 0\n20000, 20, 1\nPROBE_COMPLETE\n"
        )
        self.assertEqual(result, ("node1", [0, 2]))
        self.assertEqual(self.command[self.command.index("--time") + 1], "1:10")
        self.assertIn("+ 60 + 1", self.command[self.command.index("--wrap") + 1])

    def test_unset_power_filter_still_checks_memory(self):
        result, _ = self.run_probe(
            "20000, 300, 0\n10000, 20, 1\nPROBE_COMPLETE\n", max_power=None,
        )
        self.assertEqual(result, ("node1", [0]))

    def test_partial_window_is_rejected_with_warning(self):
        result, log = self.run_probe("20000, 20, 0\n")
        self.assertEqual(result, ("node1", []))
        self.assertIn("Incomplete GPU probe", log.call_args.args[0])

    def test_wait_timeout_or_missing_output_rejects_gpu(self):
        for options in ({"done": False}, {"output_exists": False}):
            with self.subTest(options=options):
                result, log = self.run_probe(
                    "20000, 20, 0\nPROBE_COMPLETE\n", **options,
                )
                self.assertEqual(result, ("node1", []))
                self.assertTrue(log.called)

    def test_nonpositive_duration_is_rejected(self):
        for duration in (0, -1):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                submit.probe_node_gpus(
                    "node1", "gpu", 15.0, None, duration, False, None, None, None,
                )

    def test_sampling_command_runs_full_window_and_marks_completion(self):
        self.run_probe("20000, 20, 0\nPROBE_COMPLETE\n", duration=1)
        command = self.command[self.command.index("--wrap") + 1]
        with tempfile.TemporaryDirectory() as directory:
            fake_smi = Path(directory) / "nvidia-smi"
            fake_smi.write_text("#!/bin/sh\nprintf '20000, 20, 0\\n'\n")
            fake_smi.chmod(0o755)
            environment = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}")
            start = time.monotonic()
            result = subprocess.run(
                ["sh", "-c", command], env=environment, text=True,
                capture_output=True, timeout=5, check=True,
            )
            self.assertGreaterEqual(time.monotonic() - start, 1.0)
            self.assertGreaterEqual(result.stdout.count("20000, 20, 0"), 2)
            self.assertTrue(result.stdout.endswith("PROBE_COMPLETE\n"))
            fake_smi.write_text("#!/bin/sh\nexit 1\n")
            result = subprocess.run(
                ["sh", "-c", command], env=environment, text=True,
                capture_output=True, timeout=5, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("PROBE_COMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
