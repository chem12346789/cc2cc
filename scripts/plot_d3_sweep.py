"""Append a completed D3 sweep point and redraw its progress heatmap."""

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("rs6", type=float)
    parser.add_argument("rs18", type=float)
    parser.add_argument("point_prefix", type=Path)
    parser.add_argument("--rs6-values", type=float, nargs="+", required=True)
    parser.add_argument("--rs18-values", type=float, nargs="+", required=True)
    args = parser.parse_args()

    with Path(f"{args.point_prefix}.json").open() as stream:
        best = json.load(stream)
    with Path(f"{args.point_prefix}-test.log").open() as stream:
        scores = [
            float(line.split(":", 1)[1])
            for line in stream
            if not re.search(r"warning|key|Loading|Adjusted", line, re.IGNORECASE)
            and line.startswith("After fitting:")
        ]
    if len(scores) != 1 or not np.isfinite([best["loss"], scores[0]]).all():
        raise ValueError(f"Expected finite training/test scores for {args.point_prefix}")

    results_path = args.run_dir / "results.csv"
    new_file = not results_path.exists()
    with results_path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(["rs6", "rs18", "train_loss", "test_score"])
        writer.writerow([args.rs6, args.rs18, best["loss"], scores[0]])
    with results_path.open(newline="") as stream:
        records = list(csv.DictReader(stream))

    rs6_values = np.asarray(args.rs6_values)
    rs18_values = np.asarray(args.rs18_values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    cmap = plt.cm.YlOrRd_r.copy()
    cmap.set_bad("lightgray")
    for ax, metric, title in zip(
        axes,
        ("train_loss", "test_score"),
        ("Best intermol training loss", "DIET100 score after fitting"),
    ):
        grid = np.full((len(rs18_values), len(rs6_values)), np.nan)
        for record in records:
            i = np.flatnonzero(np.isclose(rs18_values, float(record["rs18"]))).item()
            j = np.flatnonzero(np.isclose(rs6_values, float(record["rs6"]))).item()
            grid[i, j] = float(record[metric])
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
        i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.scatter(j, i, marker="*", s=180, c="white", edgecolors="black")
        ax.set_title(
            f"{title}\nmin={grid[i, j]:.6g}, "
            f"rs6={rs6_values[j]:g}, rs18={rs18_values[i]:g}"
        )
        ax.set_xlabel("rs6")
        ax.set_ylabel("rs18")
        ax.set_xticks(np.arange(len(rs6_values)), [f"{v:g}" for v in rs6_values])
        ax.set_yticks(np.arange(len(rs18_values)), [f"{v:g}" for v in rs18_values])
        fig.colorbar(image, ax=ax)
    total_points = len(rs6_values) * len(rs18_values)
    fig.suptitle(f"{args.run_dir.name}: {len(records)}/{total_points} completed points")
    fig.savefig(args.run_dir / "heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Updated {args.run_dir / 'heatmap.png'}", flush=True)


if __name__ == "__main__":
    main()
