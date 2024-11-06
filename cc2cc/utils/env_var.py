"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

STRUCTURE = os.environ.get("DFT2CC_STRUCTURE")
if STRUCTURE is None:
    STRUCTURE = "cnn3d"

LEVEL = os.environ.get("DFT2CC_LEVEL")
if LEVEL is None:
    LEVEL = 1
else:
    LEVEL = int(LEVEL)
PERIOD = os.environ.get("DFT2CC_PERIOD")
if PERIOD is None:
    PERIOD = 2
else:
    PERIOD = int(PERIOD)

ORIENTATION_NUMBER_DICT = {"x": 0, "y": 1, "z": 2}

CUBE_USE = os.environ.get("DFT2CC_CUBE_USE")
if CUBE_USE is None:
    CUBE_SIZE = 5
else:
    CUBE_SIZE = int(CUBE_USE)
CUBE_MIDDLE = CUBE_SIZE // 2
CUBE_LEN = 0.1

TEST = os.environ.get("DFT2CC_TEST")
if TEST is None:
    TEST = False

CUBE_USE = os.environ.get("DFT2CC_CUBE_USE")
if CUBE_USE is None:
    CUBE_USE = CUBE_SIZE
else:
    CUBE_USE = int(CUBE_USE)
CUBE_USE_MIDDLE = CUBE_USE // 2
ARRAY_USE_MIDDLE = CUBE_USE**3 // 2
ARRAY_USE = CUBE_USE**3

GENERATE_NEW = os.environ.get("DFT2CC_GENERATE_NEW")
if GENERATE_NEW is None:
    GENERATE_NEW = False
else:
    GENERATE_NEW = GENERATE_NEW.lower() == "true"

MAIN_PATH = os.environ.get("DFT2CC_MAIN_PATH")
if MAIN_PATH is None:
    MAIN_PATH = Path(__file__).parent.parent.parent
else:
    MAIN_PATH = Path(MAIN_PATH)

DATA_CC_PATH = os.environ.get("DFT2CC_DATA_CC_PATH")
if DATA_CC_PATH is None:
    DATA_CC_PATH = MAIN_PATH / "data" / "grids_dft"
else:
    DATA_CC_PATH = Path(DATA_CC_PATH)

DATA_PATH = os.environ.get("DFT2CC_DATA_PATH")
if DATA_PATH is None:
    DATA_PATH = MAIN_PATH / "data" / "grids_dft"
else:
    DATA_PATH = Path(DATA_PATH)

DATA_SAVE_PATH = os.environ.get("DFT2CC_DATA_SAVE_PATH")
if DATA_SAVE_PATH is None:
    DATA_SAVE_PATH = MAIN_PATH / "data" / "grids_dft" / "saved_data"
else:
    DATA_SAVE_PATH = Path(DATA_SAVE_PATH)

DATA_TEST_PATH = os.environ.get("DFT2CC_DATA_SAVE_PATH")
if DATA_TEST_PATH is None:
    DATA_TEST_PATH = MAIN_PATH / "data" / "test"
else:
    DATA_TEST_PATH = Path(DATA_TEST_PATH)

DATA_SCF_PATH = os.environ.get("DFT2CC_DATA_SCF_PATH")
if DATA_SCF_PATH is None:
    DATA_SCF_PATH = MAIN_PATH / "data" / "grids_scf"
else:
    DATA_SCF_PATH = Path(DATA_SCF_PATH)

CHECKPOINTS_PATH = os.environ.get("DFT2CC_CHECKPOINTS_PATH")
if CHECKPOINTS_PATH is None:
    CHECKPOINTS_PATH = MAIN_PATH / "checkpoints"
else:
    CHECKPOINTS_PATH = Path(CHECKPOINTS_PATH)

print(f"LEVEL: {LEVEL}")
print(f"PERIOD: {PERIOD}")
print(f"MAIN_PATH: {MAIN_PATH.resolve()}")
print(f"DATA_PATH: {DATA_PATH.resolve()}")
print(f"DATA_CC_PATH: {DATA_CC_PATH.resolve()}")
print(f"DATA_SAVE_PATH: {DATA_SAVE_PATH.resolve()}")
print(f"DATA_TEST_PATH: {DATA_TEST_PATH.resolve()}")
print(f"STRUCTURE: {STRUCTURE}")
print(f"TEST: {TEST}")
print(f"CUBE_USE: {CUBE_USE}")

if __name__ == "__main__":
    print(MAIN_PATH)
    print(DATA_PATH)
    print(DATA_SAVE_PATH)
    print(DATA_TEST_PATH)
