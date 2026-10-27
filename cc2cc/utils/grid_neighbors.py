"""Data-only prototype for center-first neighborhoods on molecular grids."""

import argparse

import numpy as np
from scipy.spatial import cKDTree


def grid_neighborhoods(coords, block_size=4096):
    """Return center + 26 nearest other entries, with indices into coords.

    Equal-distance ties have no guaranteed ordering. Coincident entries remain
    separate quadrature entries; the center itself is excluded by index.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 27:
        raise ValueError("Expected at least 27 grid coordinates with shape (N, 3).")
    if not np.isfinite(coords).all() or block_size < 1:
        raise ValueError("Coordinates must be finite and block_size positive.")
    tree = cKDTree(coords)
    neighbors = np.empty((len(coords), 27), dtype=np.int64)
    for start in range(0, len(coords), block_size):
        stop = min(start + block_size, len(coords))
        centers = np.arange(start, stop)
        distances, indices = tree.query(coords[start:stop], k=27)
        distances[indices == centers[:, None]] = np.inf
        order = np.argsort(distances, axis=1, kind="stable")[:, :26]
        neighbors[start:stop, 0] = centers
        neighbors[start:stop, 1:] = np.take_along_axis(indices, order, axis=1)
    return neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Existing NPZ containing grid coordinates.")
    parser.add_argument("--coords-key", default="coords")
    parser.add_argument("--output", required=True, help="New NPZ neighborhood archive.")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        coords = np.asarray(data[args.coords_key], dtype=np.float64)
    neighbors = grid_neighborhoods(coords)
    radius_min, radius_max, radius_sum = np.inf, 0.0, 0.0
    for start in range(0, len(coords), 4096):
        stop = min(start + 4096, len(coords))
        radius = np.linalg.norm(
            coords[neighbors[start:stop, -1]] - coords[start:stop], axis=1
        )
        radius_min = min(radius_min, radius.min())
        radius_max = max(radius_max, radius.max())
        radius_sum += radius.sum()
    # Exclusive creation prevents accidentally replacing a training archive.
    with open(args.output, "xb") as output:
        np.savez_compressed(output, coords=coords, neighbor_indices=neighbors)
    print(f"Grid entries: {len(coords)}; center-first neighborhood size: 27")
    print(f"Coordinate + index storage: {coords.nbytes + neighbors.nbytes} bytes")
    print(
        "26th-neighbor radius min/mean/max (input coordinate units): "
        f"{radius_min:.6g} / {radius_sum / len(coords):.6g} / {radius_max:.6g}"
    )


if __name__ == "__main__":
    main()
