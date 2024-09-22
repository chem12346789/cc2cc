"""
This file is used to get the environ variable.
AIDFT_MAIN_PATH: the main path of the project.
"""

from pathlib import Path
import os

CNN3D = os.environ.get("CNN3D")
if CNN3D is None:
    CNN3D = False

ORIENTATION_NUMBER_DICT = {"x": 0, "y": 1, "z": 2}
CUBE_SIZE = 9
CUBE_MIDDLE = CUBE_SIZE // 2
CUBE_USE = 3
CUBE_USE_HALF = CUBE_USE // 2
CUBE_LEN = 0.01

MAIN_PATH = os.environ.get("DFT2CC_MAIN_PATH")
if MAIN_PATH is None:
    MAIN_PATH = Path(__file__).parent.parent.parent
else:
    MAIN_PATH = Path(MAIN_PATH)

DATA_PATH = os.environ.get("DATA_PATH")
if DATA_PATH is None:
    DATA_PATH = MAIN_PATH / "data" / "grids_dft"
else:
    DATA_PATH = Path(DATA_PATH)

DATA_SAVE_PATH = os.environ.get("DATA_SAVE_PATH")
if DATA_SAVE_PATH is None:
    DATA_SAVE_PATH = MAIN_PATH / "data" / "grids_dft" / "saved_data"
else:
    DATA_SAVE_PATH = Path(DATA_SAVE_PATH)

DATA_TEST_PATH = os.environ.get("DATA_SAVE_PATH")
if DATA_TEST_PATH is None:
    DATA_TEST_PATH = MAIN_PATH / "data" / "test"
else:
    DATA_TEST_PATH = Path(DATA_TEST_PATH)

CHECKPOINTS_PATH = os.environ.get("CHECKPOINTS_PATH")
if CHECKPOINTS_PATH is None:
    CHECKPOINTS_PATH = MAIN_PATH / "checkpoints"
else:
    CHECKPOINTS_PATH = Path(CHECKPOINTS_PATH)

print(f"MAIN_PATH: {MAIN_PATH.resolve()}")
print(f"DATA_PATH: {DATA_PATH.resolve()}")
print(f"DATA_SAVE_PATH: {DATA_SAVE_PATH.resolve()}")
print(f"DATA_TEST_PATH: {DATA_TEST_PATH.resolve()}")
print(f"CNN3D: {CNN3D}")

if __name__ == "__main__":
    print(MAIN_PATH)
    print(DATA_PATH)
    print(DATA_SAVE_PATH)
    print(DATA_TEST_PATH)
