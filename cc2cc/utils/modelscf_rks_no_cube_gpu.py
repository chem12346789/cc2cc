"""Independent GPU4PySCF RKS no-cube SCF; nuclear gradients use the CPU module."""

import copy
import types

import cupy as cp
import torch
from gpu4pyscf.dft import rks
from gpu4pyscf.dft.numint import _scale_ao
from gpu4pyscf.lib.cupy_helper import tag_array

from cc2cc.utils.GridsGPU import (
    GridCube, get_cached_gdftopt, iterate_grid_segments, rho_evaluator,
)
from cc2cc.utils.grid_neighbors import grid_neighborhoods


def get_veff_modified_rks_gpu(ks, modeldict, max_memory_gpu=4000):
    """Install GPU neighbor quadrature with standard GPU4PySCF SCF assembly.

    max_memory_gpu is an integration batching budget in MB, not an allocation
    cap. The model must reside on the active CUDA device.
    """
    if modeldict.cube_type != "neighbors":
        raise ValueError("No-cube GPU RKS hooks require a neighbors model.")
    ni = copy.copy(ks._numint)

    def integrate(
        ni_, mol, grids, xc_code, dms, relativity=0, hermi=1,
        max_memory=2000, verbose=None,
    ):
        return nr_neighbors_gpu(modeldict, ni_, mol, grids, dms, max_memory_gpu)

    ni.nr_rks = types.MethodType(integrate, ni)
    ks._numint = ni
    ks.get_veff = types.MethodType(rks.get_veff, ks)


def nr_neighbors_gpu(modeldict, ni, mol, grids, dm, max_memory, *, unrestricted=False):
    """Return electron counts, XC energy and AO potential entirely on the GPU."""
    device = torch.device(modeldict.device)
    if device.type != "cuda" or (
        device.index is not None and device.index != cp.cuda.Device().id
    ):
        raise ValueError("No-cube GPU integration requires the model on the active CUDA device.")
    dm = cp.asarray(dm)
    expected = (2, mol.nao, mol.nao) if unrestricted else (mol.nao, mol.nao)
    if dm.shape != expected or cp.iscomplexobj(dm):
        raise ValueError("No-cube GPU integration requires one real RKS or UKS density.")
    if grids.coords is None:
        grids.build()
    coords = cp.asarray(grids.coords)
    weights = cp.asarray(grids.weights)
    cached = getattr(grids, "_neighbor_scf_gpu_cache", None)
    if (
        cached is None or cached[0].device.id != coords.device.id
        or not cp.array_equal(cached[0], coords)
    ):
        indices = cp.asarray(grid_neighborhoods(cp.asnumpy(coords)), dtype=cp.int64)
        cached = (coords.copy(), indices)
        grids._neighbor_scf_gpu_cache = cached
    indices = cached[1]
    opt = get_cached_gdftopt(grids, mol)
    sorted_mol = opt._sorted_mol
    sorted_dm = opt.sort_orbitals(dm, axis=[dm.ndim - 2, dm.ndim - 1])
    densities = sorted_dm if unrestricted else (sorted_dm,)
    densities = tuple(tag_array(d, mo_coeff=None, mo_occ=None) for d in densities)
    feature_dm = densities if unrestricted else densities[0]
    center_grid = copy.copy(grids)
    center_grid.opt = opt
    center_grid.cube_type = "center"
    center_grid.cube_size = 1
    center_grid.input_level = modeldict.input_level
    memory = max(1, max_memory)

    def blocks():
        start = 0
        for weight, points in iterate_grid_segments(
            sorted_mol, grids, mol.nao, 1, max_memory=memory
        ):
            stop = start + len(points)
            cube = GridCube(points[:, None, :], center_grid, sorted_mol)
            generate = cube.gen_cube_rho_uks if unrestricted else cube.gen_cube_rho_rks
            yield start, stop, weight, generate(ni, feature_dm)
            start = stop

    features = cp.empty((len(coords), modeldict.input_level), dtype=cp.float64)
    nelec = cp.zeros(len(densities), dtype=cp.float64)
    for start, stop, weight, (values, _, ao) in blocks():
        features[start:stop] = values[:, :, 0]
        for spin, density in enumerate(densities):
            rho = rho_evaluator(ni, sorted_mol, ao[0], density, xctype="LDA")
            nelec[spin] += cp.dot(weight, rho)

    derivative = cp.zeros_like(features)
    energy = cp.zeros((), dtype=cp.float64)
    batch_size = min(4096, max(1, int(memory * 1e6 / (27 * 7 * 8 * 32))))
    for start in range(0, len(coords), batch_size):
        stop = min(start + batch_size, len(coords))
        neighbors = indices[start:stop]
        unique, local = cp.unique(neighbors, return_inverse=True)
        edges = coords[neighbors] - coords[start:stop, None, :]
        energies, gradient = modeldict.eval_xc_eff(
            torch.utils.dlpack.from_dlpack(features[unique]),
            torch.utils.dlpack.from_dlpack(weights[start:stop]),
            relative_positions=torch.utils.dlpack.from_dlpack(edges),
            neighbor_indices=torch.utils.dlpack.from_dlpack(local.reshape(neighbors.shape)),
            return_tensors=True,
        )
        energy += cp.from_dlpack(energies).sum()
        derivative[unique] += cp.from_dlpack(gradient)

    potential = cp.zeros((len(densities), sorted_mol.nao, sorted_mol.nao), dtype=cp.float64)
    for start, stop, _, (_, vxc, ao) in blocks():
        if unrestricted:
            weighted = cp.einsum("islpC,pi->slp", vxc, derivative[start:stop])
        else:
            weighted = cp.einsum("ixpC,pi->xp", vxc, derivative[start:stop])[None]
        for spin, wv in enumerate(weighted):
            wv[0] *= 0.5
            aow = _scale_ao(ao, wv)
            potential[spin] += cp.dot(ao[0], aow.T)
    potential += potential.transpose(0, 2, 1).copy()
    potential = opt.unsort_orbitals(potential, axis=[1, 2])
    if unrestricted:
        return nelec, energy, potential
    return nelec[0], energy, potential[0]
