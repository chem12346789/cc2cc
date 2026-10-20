import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from scripts import install_torch


class InstallTorchTests(unittest.TestCase):
    def detect(self, cuda_version, capabilities="8.0"):
        with (
            patch.object(install_torch.shutil, "which", return_value="/bin/nvidia-smi"),
            patch.object(install_torch.subprocess, "run",
                         side_effect=[
                             Mock(stdout=f"| CUDA Version: {cuda_version} |"),
                             Mock(stdout=capabilities),
                         ]) as run,
        ):
            result = install_torch.select_backend()
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertEqual(run.call_args_list[0].args[0], ["/bin/nvidia-smi"])
        if run.call_count == 2:
            self.assertIn("--query-gpu=compute_cap", run.call_args.args[0])
        return result[0]

    def test_cuda_version_boundaries(self):
        for cuda_version, expected in (
            ("12.8", "cu128"),
            ("12.6", "cu126"),
            ("12.7", "cu126"),
            ("12.9", "cu128"),
            ("13.0", "cu128"),
        ):
            with self.subTest(cuda_version=cuda_version):
                self.assertEqual(self.detect(cuda_version), expected)

    def test_gpu_architectures(self):
        self.assertEqual(self.detect("12.8", "12.0"), "cu128")
        self.assertEqual(self.detect("12.6", "8.0\n8.6"), "cu126")
        with self.assertRaisesRegex(RuntimeError, "Blackwell"):
            self.detect("12.6", "12.0")

    def test_missing_smi_selects_cpu(self):
        with patch.object(install_torch.shutil, "which", return_value=None):
            backend, reason = install_torch.select_backend()
        self.assertEqual(backend, "cpu")
        self.assertIn("not found", reason)

    def test_unsafe_or_invalid_detection_errors(self):
        for cuda_version, capabilities in (
            ("", "8.0"), ("N/A", "8.0"), ("12.5", "8.0"),
            ("12.8", ""), ("12.8", "N/A"), ("12.8", "6.1"), ("12.8", "12.1"),
        ):
            with self.subTest(cuda_version=cuda_version, capabilities=capabilities):
                with self.assertRaises(RuntimeError):
                    self.detect(cuda_version, capabilities)

    def test_smi_failure_not_silently_converted_to_cpu(self):
        for error in (
            subprocess.CalledProcessError(1, "nvidia-smi"),
            subprocess.TimeoutExpired("nvidia-smi", 15),
        ):
            with (
                patch.object(install_torch.shutil, "which", return_value="nvidia-smi"),
                patch.object(install_torch.subprocess, "run", side_effect=error),
                self.assertRaises(subprocess.SubprocessError),
            ):
                install_torch.select_backend()

    def test_commands_use_current_python_and_matched_builds(self):
        for backend in ("cpu", "cu126", "cu128"):
            command = install_torch.install_command(backend)
            self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
            self.assertEqual(command[4:6], [
                "--index-url", f"https://download.pytorch.org/whl/{backend}"
            ])
            self.assertEqual(command[6:], [
                f"torch==2.8.0+{backend}",
                f"torchvision==0.23.0+{backend}",
                f"torchaudio==2.8.0+{backend}",
            ])

    def test_dry_run_never_invokes_pip(self):
        with (
            patch.object(install_torch, "validate_platform"),
            patch.object(install_torch, "select_backend") as detect,
            patch.object(install_torch.subprocess, "run") as run,
            patch("builtins.print"),
        ):
            self.assertEqual(install_torch.main(["--backend", "cpu", "--dry-run"]), 0)
        detect.assert_not_called()
        run.assert_not_called()

    def test_install_runs_checked_command(self):
        with (
            patch.object(install_torch, "validate_platform"),
            patch.object(install_torch, "select_backend", return_value=("cu126", "GPU")),
            patch.object(install_torch.subprocess, "run") as run,
            patch("builtins.print"),
        ):
            self.assertEqual(install_torch.main([]), 0)
        run.assert_called_once_with(install_torch.install_command("cu126"), check=True)

    def test_install_failure_is_nonzero(self):
        with (
            patch.object(install_torch, "validate_platform"),
            patch.object(install_torch.subprocess, "run",
                         side_effect=subprocess.CalledProcessError(1, "pip")),
            patch("builtins.print"),
            patch("sys.stderr"),
            self.assertRaises(SystemExit) as raised,
        ):
            install_torch.main(["--backend", "cpu"])
        self.assertEqual(raised.exception.code, 1)

    def test_platform_guard(self):
        with (
            patch.object(install_torch.platform, "system", return_value="Darwin"),
            self.assertRaisesRegex(RuntimeError, "Linux/Windows"),
        ):
            install_torch.validate_platform()


if __name__ == "__main__":
    unittest.main()
