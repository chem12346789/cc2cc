import arrow
import subprocess
from pathlib import Path

main_dir = Path(__file__).resolve().parents[0]

critical_time = arrow.now().shift(hours=-72)
for item in Path(main_dir.parents[0] / "log").glob("*"):
    if item.is_file():
        ITEM_TIME = arrow.get(item.stat().st_mtime)
        if ITEM_TIME < critical_time:
            print(str(item.absolute()))
            # remove it
            item.unlink()

with open(main_dir / "out_mkdir", "r") as outfile:
    for line in outfile:
        cmd = r"scancel {}".format(line.split()[-1])
        with open("out", "w") as outfile:
            result = subprocess.call(cmd, shell=True, stdout=outfile)
