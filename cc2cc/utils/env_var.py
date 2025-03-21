"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

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
