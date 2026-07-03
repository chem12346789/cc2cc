"""Runtime machine-info helpers.

Separated from ``env_var`` so path/constants can be imported without heavy deps.
"""

import os

from cc2cc.utils.env_var import (
    CHECKPOINTS_PATH,
    CUBE_MIDDLE,
    DATA_PATH,
    DATA_TEST_PATH,
    EDGE_LEN,
    EDGE_SIZE,
    MAIN_PATH,
    TEST,
)


def print_computer_info(device):
    """Print CPU/GPU runtime info and project environment settings."""
    from pyscf import lib
    import numba
    import torch

    print(f"PID: {os.getpid()}")
    print(f"CPU Count: {os.cpu_count()}")
    omp_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
    print(f"OMP_NUM_THREADS: {omp_num_threads}")
    print(f"NUMBER_OF_GPU: {os.environ.get('NUMBER_OF_GPU', 'Not Set')}")
    lib.num_threads(omp_num_threads)
    torch.set_num_threads(omp_num_threads)
    numba.set_num_threads(omp_num_threads)

    if device == "cuda":
        print(f"Is Available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU number: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Number of Devices: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

    print(f"EDGE_SIZE: {EDGE_SIZE}")
    print(f"CUBE_MIDDLE: {CUBE_MIDDLE}")
    print(f"EDGE_LEN: {EDGE_LEN}")
    print(f"TEST: {TEST}")
    print(f"MAIN_PATH: {MAIN_PATH.resolve()}")
    print(f"DATA_PATH: {DATA_PATH.resolve()}")
    print(f"DATA_TEST_PATH: {DATA_TEST_PATH.resolve()}")
    print(f"CHECKPOINTS_PATH: {CHECKPOINTS_PATH.resolve()}")
