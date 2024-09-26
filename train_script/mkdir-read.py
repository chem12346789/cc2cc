import time
import subprocess
from pathlib import Path
import itertools
import arrow


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
template_bash = main_dir / "train-template.bash"
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

work_dir = main_dir / ("bash_submitted" + time_stamp)
work_dir.mkdir()
work_bash = work_dir / "train-template.bash"

LIST_OF_GPU = itertools.cycle([0, 1])

for (
    batch_size,
    eval_step,
    load_model,
    with_eval,
    structure,
    cube_use,
) in itertools.product(
    [2**15],  # batch_size
    [10],  # eval_step
    ["New"],  # load_model
    ["False"],  # with_eval
    [
        # "cnn3d",
        # "fc_3d",
        # "fc",
        "unet",
    ],  # structure
    [3],  # cube_use
):
    number_of_gpu = next(LIST_OF_GPU)
    cmd = f"""cp {template_bash} {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_EVAL_STEP/{eval_step}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_BATCH_SIZE/{batch_size}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_WITH_EVAL/{with_eval}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_LOAD_MODEL/{load_model}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_STRUCTURE/{structure}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_NUMBER_OF_GPU/{number_of_gpu}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_CUBE_USE/{cube_use}/g" {work_bash}"""
    cmd += (
        "&&"
        + f"""mv {work_bash} {work_dir / f"train_{eval_step}_{batch_size}_{with_eval}_{load_model}_{structure}_{cube_use}.bash"}"""
    )
    with open(main_dir / "out_mkdir", "w", encoding="utf-8") as f:
        subprocess.call(cmd, shell=True, stdout=f)

for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd = f"""sbatch < {child}"""
        with open(main_dir / "out_mkdir", "a", encoding="utf-8") as f:
            subprocess.call(cmd, shell=True, stdout=f)

        time.sleep(6 * 6)
