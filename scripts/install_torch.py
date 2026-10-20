"""Install a matched PyTorch stack into the interpreter running this script."""

import argparse
import platform
import re
import shlex
import shutil
import subprocess
import sys


VERSIONS = {"torch": "2.8.0", "torchvision": "0.23.0", "torchaudio": "2.8.0"}


def validate_platform():
    if platform.system() not in {"Linux", "Windows"} or platform.machine().lower() not in {
        "x86_64", "amd64"
    }:
        raise RuntimeError("This installer supports Linux/Windows x86_64 only.")
    if not (3, 9) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("This PyTorch stack requires Python 3.9 through 3.13.")


def select_backend():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "cpu", "nvidia-smi not found; selecting CPU (use --backend on login nodes)."
    result = subprocess.run(
        [executable], capture_output=True, text=True, check=True, timeout=15,
    )
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if match is None:
        raise RuntimeError("Cannot find CUDA Version in nvidia-smi output.")
    cuda_version = tuple(int(part) for part in match.groups())
    if cuda_version >= (12, 8):
        backend = "cu128"
    elif cuda_version >= (12, 6):
        backend = "cu126"
    else:
        raise RuntimeError(
            f"nvidia-smi reports CUDA {match.group(1)}.{match.group(2)}; "
            "this stack requires CUDA support >= 12.6. "
            "Upgrade the driver or choose --backend cpu explicitly."
        )
    result = subprocess.run(
        [executable, "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs; use --backend cpu explicitly.")
    try:
        capabilities = [tuple(int(part) for part in row.split(".")) for row in rows]
    except ValueError as error:
        raise RuntimeError(f"Cannot parse nvidia-smi GPU metadata: {rows!r}") from error
    if any(not (7, 5) <= cap <= (12, 0) for cap in capabilities):
        raise RuntimeError(
            "Auto-selection supports compute capability 7.5 through 12.0 only; "
            "check GPU support and choose --backend explicitly."
        )
    if backend == "cu126" and any(cap >= (10, 0) for cap in capabilities):
        raise RuntimeError("Blackwell GPUs require cu128 here; upgrade the driver.")
    return backend, (
        f"nvidia-smi CUDA Version: {cuda_version[0]}.{cuda_version[1]}; "
        f"GPU compute capabilities: {rows!r}; selecting {backend}."
    )


def install_command(backend):
    return [
        sys.executable, "-m", "pip", "install",
        "--index-url", f"https://download.pytorch.org/whl/{backend}",
        *(f"{name}=={version}+{backend}" for name, version in VERSIONS.items()),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("auto", "cpu", "cu126", "cu128"), default="auto",
        help="Explicit choices bypass GPU detection; verify target GPU/driver compatibility.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the selection and command without invoking pip or using the network.",
    )
    args = parser.parse_args(argv)
    try:
        validate_platform()
        if args.backend == "auto":
            backend, reason = select_backend()
        else:
            backend, reason = args.backend, f"Explicit backend: {args.backend}."
        command = install_command(backend)
        print(reason, flush=True)
        print(f"Interpreter: {sys.executable}", flush=True)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Installation failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
