"""Self-contained CPU UKS neighborhood quadrature and nuclear-gradient hooks."""

import copy
import types

import numpy as np
from pyscf import lib
from pyscf.dft import rks, uks
from pyscf.dft.numint import _dot_ao_ao_dense, _scale_ao
from pyscf.grad.rks import _gga_grad_sum_
from pyscf.grad import rks as rks_grad, uks as uks_grad
from scipy.spatial import cKDTree

from cc2cc.utils.Grids import GridCube, iterate_grid_segments, iterate_grid_response_segments
from cc2cc.utils.grid_neighbors import grid_neighborhoods


def _install_scf(ks, modeldict, unrestricted):
    if modeldict.cube_type != "neighbors":
        raise ValueError("No-cube SCF hooks require a neighbors model.")
    ni = copy.copy(ks._numint)

    def integrate(
        ni_, mol, grids, xc_code, dms, relativity=0, hermi=1,
        max_memory=2000, verbose=None,
    ):
        return nr_neighbors(
            modeldict, ni_, mol, grids, dms, max_memory,
            unrestricted=unrestricted,
        )

    setattr(ni, "nr_uks" if unrestricted else "nr_rks", types.MethodType(integrate, ni))
    ks._numint = ni
    ks.get_veff = types.MethodType((uks if unrestricted else rks).get_veff, ks)


def get_veff_modified_uks(ks, modeldict):
    """Install CPU no-cube UKS quadrature with standard PySCF energy assembly."""
    _install_scf(ks, modeldict, unrestricted=True)


def _install_gradient(ks_grad, modeldict, unrestricted):
    if modeldict.cube_type != "neighbors":
        raise ValueError("No-cube gradient hooks require a neighbors model.")
    standard_get_veff = ks_grad.get_veff
    grad = uks_grad if unrestricted else rks_grad

    def get_veff(gradient, mol=None, dm=None):
        if mol is None:
            mol = gradient.mol
        if dm is None:
            dm = gradient.base.make_rdm1()
        # Preserve the original gradient class's J/K, NLC and DF auxiliary response.
        standard = standard_get_veff(mol, dm)
        mf = gradient.base
        grids = gradient.grids if gradient.grids is not None else mf.grids
        memory = max(1, gradient.max_memory * 0.9 - lib.current_memory()[0])
        evaluate = grad.get_vxc_full_response if gradient.grid_response else grad.get_vxc
        baseline_response, baseline_potential = evaluate(
            mf._numint, mol, grids, mf.xc, dm, max_memory=memory,
        )
        response, potential = nr_neighbors(
            modeldict, mf._numint, mol, grids, dm, memory,
            unrestricted=unrestricted, nuclear_gradient=True,
            grid_response=gradient.grid_response,
        )
        result = lib.tag_array(
            standard + potential - baseline_potential, **standard.__dict__
        )
        if gradient.grid_response:
            result.exc1_grid = standard.exc1_grid + response - baseline_response
        return result

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)


def get_veff_grad_modified_uks(ks_grad, modeldict):
    """Install no-cube UKS gradients with full model grid response."""
    _install_gradient(ks_grad, modeldict, unrestricted=True)


def nr_neighbors(
    modeldict, ni, mol, grids, dm, max_memory, *, unrestricted=True,
    nuclear_gradient=False, grid_response=False,
):
    """Evaluate XC potentials or nuclear terms with optional full grid response.

    Neighbor/pruning selections are held fixed on each smooth branch.
    PySCF assembles these XC terms with core, J/K, nuclear and Pulay terms.
    """
    if grid_response and not nuclear_gradient:
        raise ValueError("Grid response requires nuclear_gradient=True.")
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

    def blocks(ao_deriv=1):
        start = 0
        for _, weights, coords in iterate_grid_segments(
            mol, grids, mol.nao, ao_deriv, max_memory=memory
        ):
            stop = start + len(coords)
            cube = GridCube(coords[:, None, :], center_grid)
            generate = cube.gen_cube_rho_uks if unrestricted else cube.gen_cube_rho_rks
            yield start, stop, weights, generate(ni, dm, ao_deriv=ao_deriv)
            start = stop

    for start, stop, weights, (values, _, ao) in blocks():
        features[start:stop] = values[:, :, 0]
        densities = dm if unrestricted else (dm,)
        for spin, density in enumerate(densities):
            rho = ni.eval_rho(mol, ao[0], density, xctype="LDA")
            nelec[spin] += np.dot(weights, rho)

    feature_gradient = np.zeros_like(features)
    if grid_response:
        weight_gradient = np.zeros(len(features), dtype=np.float64)
        position_gradient = np.zeros((len(features), 3), dtype=np.float64)
    energy = 0.0
    # Bound gathered features, geometry and network batches independently of AO blocks.
    batch_size = min(4096, max(1, int(memory * 1e6 / (27 * 7 * 8 * 32))))
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        neighbors = indices[start:stop]
        unique, local_indices = np.unique(neighbors, return_inverse=True)
        local_indices = local_indices.reshape(neighbors.shape)
        values = features[unique]
        edges = grids.coords[neighbors] - grids.coords[start:stop, None, :]
        outputs = modeldict.eval_xc_eff(
            values, grids.weights[start:stop], relative_positions=edges,
            neighbor_indices=local_indices,
            **({"weight_response": True, "position_response": True} if grid_response else {}),
        )
        energies, derivative = outputs[:2]
        if grid_response:
            weight_gradient[start:stop] = outputs[2]
            edges_gradient = outputs[3]
            np.add.at(position_gradient, neighbors.ravel(), edges_gradient.reshape(-1, 3))
            position_gradient[start:stop] -= edges_gradient.sum(axis=1)
        energy += energies.sum()
        # Unique within this cluster; overlapping clusters still accumulate.
        feature_gradient[unique] += derivative

    if grid_response:
        return _moving_grid_terms(
            ni, mol, grids, dm, memory, center_grid, feature_gradient,
            weight_gradient, position_gradient, unrestricted,
        )

    nspin = 2 if unrestricted else 1
    shape = (
        (nspin, 3, mol.nao, mol.nao) if nuclear_gradient
        else (nspin, mol.nao, mol.nao)
    )
    potential = np.zeros(shape, dtype=np.float64)
    for start, stop, _, (_, vxc, ao) in blocks(2 if nuclear_gradient else 1):
        middle = feature_gradient[start:stop]
        if unrestricted:
            weighted = np.einsum("islpC,pi->slp", vxc, middle, optimize=True)
        else:
            weighted = np.einsum("ixpC,pi->xp", vxc, middle, optimize=True)[None]
        for spin, wv in enumerate(weighted):
            wv[0] *= 0.5
            if nuclear_gradient:
                _gga_grad_sum_(potential[spin], mol, ao, wv, None, mol.ao_loc_nr())
            else:
                _dot_ao_ao_dense(ao[0], _scale_ao(ao, wv), None, potential[spin])
    if nuclear_gradient:
        if unrestricted:
            return np.zeros((mol.natm, 3)), -potential
        return None, -potential[0]
    potential = lib.hermi_sum(potential, axes=(0, 2, 1))
    if unrestricted:
        return nelec, energy, potential
    return nelec[0], energy, potential[0]


def _moving_grid_terms(
    ni, mol, grids, dm, memory, center_grid, feature_gradient,
    weight_gradient, position_gradient, unrestricted,
):
    """Assign sampling and edge motion to the atoms owning the sampled points."""
    tree = cKDTree(grids.coords)
    visited = np.zeros(len(grids.coords), dtype=bool)
    densities = dm if unrestricted else (dm,)
    potential = np.zeros((len(densities), 3, mol.nao, mol.nao), dtype=np.float64)
    response = np.zeros((mol.natm, 3), dtype=np.float64)

    def accumulate(selected, coords, atom=None):
        cube = GridCube(coords[:, None, :], center_grid)
        generate = cube.gen_cube_rho_uks if unrestricted else cube.gen_cube_rho_rks
        _, vxc, ao = generate(ni, dm, ao_deriv=2)
        middle = feature_gradient[selected]
        if unrestricted:
            weighted = np.einsum("islpC,pi->slp", vxc, middle, optimize=True)
        else:
            weighted = np.einsum("ixpC,pi->xp", vxc, middle, optimize=True)[None]
        for spin, wv in enumerate(weighted):
            wv[0] *= 0.5
            block = np.zeros((3, mol.nao, mol.nao), dtype=np.float64)
            _gga_grad_sum_(block, mol, ao, wv, None, mol.ao_loc_nr())
            potential[spin] += block
            if atom is not None:
                response[atom] += 2 * np.einsum("xij,ji->x", block, densities[spin])

    for atom, _, weights, coords, weight1 in iterate_grid_response_segments(
        mol, grids, mol.nao, 2, memory
    ):
        distance, indices = tree.query(coords)
        keep = distance < 1e-12
        selected = indices[keep]
        if not len(selected):
            continue
        if visited[selected].any() or len(np.unique(selected)) != len(selected):
            raise NotImplementedError("Grid response requires distinct physical grid coordinates.")
        if not np.allclose(weights[keep], grids.weights[selected], rtol=1e-10, atol=1e-14):
            raise ValueError("SCF weights do not match the atom-centered response grid.")
        visited[selected] = True
        accumulate(selected, coords[keep], atom)
        response += np.einsum("axp,p->ax", weight1[:, :, keep], weight_gradient[selected])
        response[atom] += position_gradient[selected].sum(axis=0)
    # PySCF padding stays at (1e-4, 1e-4, 1e-4) as nuclei move. Halo
    # features still depend on moving AOs, but padding has no grid motion.
    padding = (~visited & (grids.weights == 0)
               & np.all(grids.coords == 1e-4, axis=1))
    selected = np.flatnonzero(padding)
    if len(selected):
        accumulate(selected, grids.coords[selected])
    unmatched = ~visited & ~padding
    if (np.any(grids.weights[unmatched] != 0)
            or np.any(feature_gradient[unmatched] != 0)
            or np.any(position_gradient[unmatched] != 0)):
        raise ValueError(
            "SCF grid contains active points without an atom-centered response "
            f"({np.count_nonzero(unmatched)} unmatched non-padding points)."
        )
    return response, -potential if unrestricted else -potential[0]
