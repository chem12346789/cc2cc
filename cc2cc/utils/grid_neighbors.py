"""Data-only prototype for center-first neighborhoods on molecular grids."""

import argparse

import numpy as np
from scipy.spatial import cKDTree


class RadiusNeighborhoods:
    """Lazy center-first radius queries; no full edge list is retained."""

    def __init__(self, coords, radius):
        self.coords = np.asarray(coords, dtype=np.float64)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("Expected grid coordinates with shape (N, 3).")
        if not np.isfinite(self.coords).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Radius and grid coordinates must be finite; radius must be positive.")
        self.radius = radius
        self.tree = cKDTree(self.coords)
        self.counts = self.tree.query_ball_point(self.coords, radius, return_length=True)

    def ranges(self, max_entries):
        start = 0
        while start < len(self.coords):
            stop, width = start, 0
            while stop < len(self.coords):
                next_width = max(width, int(self.counts[stop]))
                if stop > start and (stop + 1 - start) * next_width > max_entries:
                    break
                width = next_width
                stop += 1
            yield start, stop
            start = stop

    def batches(self, max_entries):
        for start, stop in self.ranges(max_entries):
            yield start, stop, self.query(start, stop)

    def query(self, start, stop):
        centers = np.arange(start, stop)
        width = int(self.counts[start:stop].max())
        indices = np.broadcast_to(centers[:, None], (stop - start, width)).copy()
        rows = self.tree.query_ball_point(self.coords[start:stop], self.radius)
        for row, (center, neighbors) in enumerate(zip(centers, rows)):
            neighbors = [index for index in neighbors if index != center]
            indices[row, 1:len(neighbors) + 1] = neighbors
        return indices


def build_neighborhoods(coords, model):
    radius = getattr(model, "radius_cutoff", None)
    if radius is not None:
        return RadiusNeighborhoods(coords, radius)
    return grid_neighborhoods(coords, neighborhood_size=model.cube_size)


def neighborhoods_match(indices, model):
    radius = getattr(model, "radius_cutoff", None)
    if isinstance(indices, RadiusNeighborhoods):
        return indices.radius == radius
    return radius is None and indices.shape[1] == model.cube_size


def neighborhood_batches(indices, batch_size, neighborhood_size):
    if isinstance(indices, RadiusNeighborhoods):
        yield from indices.batches(batch_size * neighborhood_size)
    else:
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            yield start, stop, indices[start:stop]


def neighborhood_edges(coords, neighbors, start, radius=None):
    """Place padding outside the cutoff, without masking real coincident points."""
    edges = coords[neighbors] - coords[start:start + len(neighbors), None, :]
    if radius is not None:
        padding = neighbors[:, 1:] == neighbors[:, :1]
        edges[:, 1:][padding] = (2 * radius, 0, 0)
    return edges


def grid_neighborhoods(coords, block_size=4096, *, neighborhood_size=27):
    """Return center-first neighborhoods, with indices into coords.

    Equal-distance ties have no guaranteed ordering. Coincident entries remain
    separate quadrature entries; the center itself is excluded by index.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if not isinstance(neighborhood_size, int) or neighborhood_size < 2:
        raise ValueError("neighborhood_size must be an integer of at least two.")
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < neighborhood_size:
        raise ValueError(
            f"Expected at least {neighborhood_size} grid coordinates with shape (N, 3)."
        )
    if not np.isfinite(coords).all() or block_size < 1:
        raise ValueError("Coordinates must be finite and block_size positive.")
    tree = cKDTree(coords)
    neighbors = np.empty((len(coords), neighborhood_size), dtype=np.int64)
    for start in range(0, len(coords), block_size):
        stop = min(start + block_size, len(coords))
        centers = np.arange(start, stop)
        distances, indices = tree.query(coords[start:stop], k=neighborhood_size)
        distances[indices == centers[:, None]] = np.inf
        order = np.argsort(distances, axis=1, kind="stable")[:, :neighborhood_size - 1]
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
