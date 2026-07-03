"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

# set the environment variable
EDGE_SIZE = os.environ.get("DFT2CC_EDGE_SIZE")
if EDGE_SIZE is None:
    EDGE_SIZE = 3
else:
    EDGE_SIZE = int(EDGE_SIZE)
CUBE_MIDDLE = EDGE_SIZE // 2

EDGE_LEN = os.environ.get("DFT2CC_EDGE_LEN")
if EDGE_LEN is None:
    EDGE_LEN = 1e-2
else:
    EDGE_LEN = float(EDGE_LEN)

TEST = os.environ.get("DFT2CC_TEST")
if TEST is None:
    TEST = False

MAIN_PATH = os.environ.get("DFT2CC_MAIN_PATH")
if MAIN_PATH is None:
    MAIN_PATH = Path(__file__).parent.parent.parent
else:
    MAIN_PATH = Path(MAIN_PATH)

DATA_DIR = os.environ.get("DFT2CC_DATA_DIR")
if DATA_DIR is None:
    DATA_PATH = MAIN_PATH / "data" / "grids_dft"
else:
    DATA_PATH = MAIN_PATH / "data" / DATA_DIR

DATA_TEST_DIR = os.environ.get("DFT2CC_DATA_TEST_DIR")
if DATA_TEST_DIR is None:
    DATA_TEST_PATH = MAIN_PATH / "data" / "test"
else:
    DATA_TEST_PATH = MAIN_PATH / "data" / DATA_TEST_DIR

CHECKPOINTS_PATH = MAIN_PATH / "checkpoints"
