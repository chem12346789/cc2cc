"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

import torch


def print_gpu_info(device):
    """
    Print information about the GPU and CUDA environment.

    This function prints:
    - Current process ID
    - CUDA availability
    - GPU device name
    - CUDA visible devices from environment
    - Current CUDA device
    - Number of available CUDA devices

    If PyTorch is not installed, it will display a message instead.
    """
    # print the information of the process
    print(f"PID: {os.getpid()}")

    if device == "cuda":
        # print the gpu information
        print(f"Is Available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU number: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Number of Devices: {torch.cuda.device_count()}")


# set the environment variable

LEVEL = 3
PERIOD = 2

CUBE_SIZE = os.environ.get("DFT2CC_CUBE_SIZE")
if CUBE_SIZE is None:
    CUBE_SIZE = 3
else:
    CUBE_SIZE = int(CUBE_SIZE)
CUBE_MIDDLE = CUBE_SIZE // 2
CUBE_LEN = 0.1

TEST = os.environ.get("DFT2CC_TEST")
if TEST is None:
    TEST = False

MAIN_PATH = os.environ.get("DFT2CC_MAIN_PATH")
if MAIN_PATH is None:
    MAIN_PATH = Path(__file__).parent.parent.parent
else:
    MAIN_PATH = Path(MAIN_PATH)

DATA_PATH = MAIN_PATH / "data" / "grids_dft"
DATA_TEST_PATH = MAIN_PATH / "data" / "test"
DATA_SCF_PATH = MAIN_PATH / "data" / "grids_scf"
CHECKPOINTS_PATH = MAIN_PATH / "checkpoints"

GENERATE_DATA = os.environ.get("DFT2CC_GENERATE_DATA")
if GENERATE_DATA is None:
    GENERATE_DATA = False

if __name__ == "__main__":
    print(f"MAIN_PATH: {MAIN_PATH.resolve()}")
    print(f"DATA_PATH: {DATA_PATH.resolve()}")
    print(f"DATA_TEST_PATH: {DATA_TEST_PATH.resolve()}")
    print(f"TEST: {TEST}")
