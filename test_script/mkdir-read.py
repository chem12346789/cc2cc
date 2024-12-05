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
template_bash = main_dir / "validate-template.bash"
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
work_bash = work_dir / "validate-template.bash"

gpu_node_pool = itertools.cycle(
    [
        # "gpu03",
        # "gpu03",
        # "gpu07",
        # "gpu07",
        "gpu04",
        "gpu04",
        "gpu04",
        "gpu04",
        "gpu05",
        "gpu05",
        "gpu05",
        "gpu05",
        "gpu06",
        "gpu06",
        "gpu06",
        "gpu06",
    ]
)


for (
    checkpoint_hidden_size,
    cube_use,
    (range_list, extend_atom),
) in itertools.product(
    [
        # "checkpoint-ccdft_2024-09-27-14-27-34",
        "checkpoint-ccdft_2024-09-29-18-16-08",
    ],  # checkpoint_hidden_size
    [3],  # cube_use
    [
        ((-0.5, -0.5, 1), "0-1"),
        ((-0.4, -0.4, 1), "0-1"),
        ((-0.3, -0.3, 1), "0-1"),
        ((-0.2, -0.2, 1), "0-1"),
        ((-0.1, -0.1, 1), "0-1"),
        ((0, 0, 1), "0"),
        ((0.1, 0.1, 1), "0-1"),
        ((0.2, 0.2, 1), "0-1"),
        ((0.3, 0.3, 1), "0-1"),
        ((0.4, 0.4, 1), "0-1"),
        ((0.5, 0.5, 1), "0-1"),
    ],
):
    (_, checkpoint) = checkpoint_hidden_size.split("_")
    print(checkpoint)
    cmd = f"""cp {template_bash} {work_bash}"""
    gpu_node = next(gpu_node_pool)
    cmd += "&&" + f"""sed -i "s/BASH_GPU_NODE/{gpu_node}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/BASH_CUBE_USE/{cube_use}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/CHECKPOINT/{checkpoint}/g" {work_bash}"""
    cmd += "&&" + f"""sed -i "s/EXTEND_ATOM/{extend_atom}/g" {work_bash}"""

    if isinstance(range_list, float):
        start = range_list
        cmd += "&&" + f"""sed -i "s/BASH_VALIDATE_NAME/{start}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/START/{start}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/END//g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/STEP//g" {work_bash}"""
        cmd += (
            "&&"
            + f"""mv {work_bash} {work_dir / f"validate_{checkpoint_hidden_size}_{start}.bash"}"""
        )
    elif isinstance(range_list, tuple):
        start = range_list[0]
        end = range_list[1]
        step = range_list[2]
        cmd += "&&" + f"""sed -i "s/BASH_VALIDATE_NAME/{start}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/START/{start}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/END/{end}/g" {work_bash}"""
        cmd += "&&" + f"""sed -i "s/STEP/{step}/g" {work_bash}"""
        cmd += (
            "&&"
            + f"""mv {work_bash} {work_dir / f"validate_{checkpoint_hidden_size}_{start}_{end}_{step}.bash"}"""
        )

    with open(main_dir / "out_mkdir", "w", encoding="utf-8") as f:
        subprocess.call(cmd, shell=True, stdout=f)

for child in (work_dir).glob("*.bash"):
    if child.is_file():
        cmd = f"""sbatch < {child}"""
        with open(main_dir / "out_mkdir", "a", encoding="utf-8") as f:
            subprocess.call(cmd, shell=True, stdout=f)
        time.sleep(1)
