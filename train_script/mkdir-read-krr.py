import time
import subprocess
from pathlib import Path
import itertools

import arrow
import numpy as np


def clean_dir(pth):
    """
    clean the directory
    """
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            clean_dir(child)
            child.rmdir()


main_dir = Path(__file__).resolve().parents[0]
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

critical_time = arrow.now().shift(hours=-72)
for item in Path(main_dir).glob("*"):
    if not item.is_file():
        ITEM_TIME = arrow.get(item.stat().st_mtime)
        if ITEM_TIME < critical_time:
            # remove it
            print(str(item.absolute()))
            clean_dir(item)
            item.rmdir()

# renew out_mkdir
if (main_dir / "out_mkdir").exists():
    (main_dir / "out_mkdir").unlink()
(main_dir / "out_mkdir").touch()

template_bash = main_dir / "train-template-krr.bash"
work_dir = main_dir / ("bash_submitted" + time_stamp)
work_dir.mkdir()
work_bash = work_dir / "train-template-krr.bash"

for (
    gamma,
    alpha,
) in itertools.product(
    10 ** np.linspace(-1, 3, 5)[::-1],  # gamma
    [0.01, 0.1, 1],  # alpha
):
    cmd = f"""cp {template_bash} {work_bash}"""
    cmd += "&&" + f"""sed -i 's/BASH_GAMMA/{gamma}/g' {work_bash}"""
    cmd += "&&" + f"""sed -i 's/BASH_ALPHA/{alpha}/g' {work_bash}"""
    cmd += "&&" + f"""mv {work_bash} {work_dir / f"train_{gamma}_{alpha}.bash"}"""
    with open(main_dir / "out_mkdir", "w", encoding="utf-8") as f:
        subprocess.call(cmd, shell=True, stdout=f)

for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd = f"""sbatch < {child}"""
        with open(main_dir / "out_mkdir", "a", encoding="utf-8") as f:
            subprocess.call(cmd, shell=True, stdout=f)

        time.sleep(1)
