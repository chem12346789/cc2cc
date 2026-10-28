"""Shared CPU quadrature for neighborhood-model RKS/UKS SCF."""

import copy

import numpy as np
from pyscf import lib
from pyscf.dft.numint import _dot_ao_ao_dense, _scale_ao

from cc2cc.utils.Grids import GridCube, iterate_grid_segments
from cc2cc.utils.grid_neighbors import grid_neighborhoods


def nr_neighbors(modeldict, ni, mol, grids, dm, max_memory, *, unrestricted=False):
    """Evaluate semilocal baseline + nonlocal neighborhood correction once per grid."""
    dm = np.asarray(dm)
    if dm.shape != ((2, mol.nao, mol.nao) if unrestricted else (mol.nao, mol.nao)):
        raise ValueError("No-cube SCF requires one real RKS or spin-resolved UKS density.")
    if np.iscomplexobj(dm):
        raise NotImplementedError("Complex no-cube densities are not supported.")
    if grids.coords is None:
        grids.build(with_non0tab=False, sort_grids=False)
    cached = getattr(grids, "_neighbor_scf_cache", None)
    if cached is None or not np.array_equal(cached[0], grids.coords):
        cached = (np.array(grids.coords, copy=True), grid_neighborhoods(grids.coords))
        grids._neighbor_scf_cache = cached
    indices = cached[1]
    center_grid = copy.copy(grids)
    center_grid.cube_type = "center"
    center_grid.cube_size = 1
    center_grid.input_level = modeldict.input_level
    features = np.empty((len(grids.coords), modeldict.input_level), dtype=np.float64)
    nelec = np.zeros(2 if unrestricted else 1, dtype=np.float64)
    memory = max(1, max_memory)

    def blocks():
        start = 0
        for _, weights, coords in iterate_grid_segments(
            mol, grids, mol.nao, 1, max_memory=memory
        ):
            stop = start + len(coords)
            cube = GridCube(coords[:, None, :], center_grid)
            generate = cube.gen_cube_rho_uks if unrestricted else cube.gen_cube_rho_rks
            yield start, stop, weights, generate(ni, dm, ao_deriv=1)
            start = stop

    for start, stop, weights, (values, _, ao) in blocks():
        features[start:stop] = values[:, :, 0]
        densities = dm if unrestricted else (dm,)
        for spin, density in enumerate(densities):
            rho = ni.eval_rho(mol, ao[0], density, xctype="LDA")
            nelec[spin] += np.dot(weights, rho)

    feature_gradient = np.zeros_like(features)
    energy = 0.0
    # Bound gathered features, geometry and network batches independently of AO blocks.
    batch_size = min(4096, max(1, int(memory * 1e6 / (27 * 7 * 8 * 32))))
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        neighbors = indices[start:stop]
        values = features[neighbors].transpose(0, 2, 1)
        edges = grids.coords[neighbors] - grids.coords[start:stop, None, :]
        energies, derivative = modeldict.eval_xc_eff(
            values, grids.weights[start:stop], relative_positions=edges
        )
        energy += energies.sum()
        np.add.at(
            feature_gradient, neighbors.reshape(-1),
            derivative.transpose(0, 2, 1).reshape(-1, modeldict.input_level),
        )

    potential = np.zeros((2 if unrestricted else 1, mol.nao, mol.nao), dtype=np.float64)
    for start, stop, _, (_, vxc, ao) in blocks():
        middle = feature_gradient[start:stop]
        if unrestricted:
            weighted = np.einsum("islpC,pi->slp", vxc, middle, optimize=True)
        else:
            weighted = np.einsum("ixpC,pi->xp", vxc, middle, optimize=True)[None]
        for spin, wv in enumerate(weighted):
            wv[0] *= 0.5
            _dot_ao_ao_dense(ao[0], _scale_ao(ao, wv), None, potential[spin])
    potential = lib.hermi_sum(potential, axes=(0, 2, 1))
    if unrestricted:
        return nelec, energy, potential
    return nelec[0], energy, potential[0]
