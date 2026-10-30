"""Fixed-reference orbital and compact observable constraints for generation."""

from pathlib import Path

import numpy as np

from cc2cc.utils.env_var import EDGE_LEN


def _observable_operators(mol, grids, ni, block_size):
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    with mol.with_common_orig(np.zeros(3, dtype=np.float64)):
        dipole = -mol.intor_symmetric("int1e_r", comp=3)
        second = mol.intor("int1e_rr", comp=9).reshape(3, 3, mol.nao, mol.nao)
    projections = np.zeros((mol.natm, mol.nao, mol.nao), dtype=np.float64)
    centers = mol.atom_coords()
    for start in range(0, len(grids.coords), block_size):
        stop = start + block_size
        coords = grids.coords[start:stop]
        ao = ni.eval_ao(mol, coords)
        gaussian = np.exp(-np.sum((centers[:, None] - coords) ** 2, axis=2))
        projections += np.einsum(
            "gp,g,gq,kg->kpq", ao, grids.weights[start:stop], ao, gaussian,
            optimize=True,
        )
    operators = np.concatenate(
        (dipole, np.array([second[i, j] for i, j in pairs]), projections)
    )
    labels = ["dipole_x", "dipole_y", "dipole_z"]
    labels += ["moment_" + "xyz"[i] + "xyz"[j] for i, j in pairs]
    labels += [f"atom_gaussian_{i}" for i in range(mol.natm)]
    nuclear = np.zeros(len(labels), dtype=np.float64)
    nuclear[:3] = mol.atom_charges() @ centers
    return operators, np.array(labels), nuclear


def get_zmp_constraint_data(mf, grids, dm_zmp, dm_target, cache_prefix, args):
    """Generate optional float64 caches without changing the reference density.

    Sidecars have axes (center, feature, cube_sample, virtual, occupied) for
    orbital Jacobians and (center, feature, cube_sample, potential) for compact
    projection Jacobians. UKS uses a separate sidecar for each spin.
    """
    feature_ov = getattr(args, "zmp_feature_ov", True)
    response = getattr(args, "zmp_observable_response", True)
    if not (feature_ov or response):
        return {}
    block_size = getattr(args, "zmp_response_block_size", 32)
    gap_floor = getattr(args, "zmp_response_gap_floor", 1e-3)
    if block_size <= 0 or gap_floor <= 0:
        raise ValueError("ZMP response block size and gap floor must be positive.")
    mol = mf.mol
    overlap = np.asarray(mf.get_ovlp(), dtype=np.float64)
    density_energy, coeff = mf.eig(-overlap @ dm_zmp @ overlap, overlap)
    occ = mf.get_occ(density_energy, coeff)
    fock = mf.get_hcore() + mf.get_veff(mol, dm_zmp)
    restricted = dm_zmp.ndim == 2
    coeffs = np.array([coeff] if restricted else coeff, dtype=np.float64)
    occs = np.array([occ] if restricted else occ, dtype=np.float64)
    focks = np.array([fock] if restricted else fock, dtype=np.float64)
    densities = np.array([dm_zmp] if restricted else dm_zmp, dtype=np.float64)
    energies = np.zeros_like(occs)
    for spin, (c, n, f, dm, e) in enumerate(zip(coeffs, occs, focks, densities, energies)):
        if not np.allclose((c * n) @ c.T, dm, rtol=0, atol=1e-7):
            raise ValueError("ZMP response requires an integer-occupation determinant.")
        for selected in (n > 0, n == 0):
            block = c[:, selected]
            e[selected], rotation = np.linalg.eigh(block.T @ f @ block)
            c[:, selected] = block @ rotation
        if response and np.any(e[n == 0, None] - e[None, n > 0] <= 0):
            raise ValueError(
                f"Nonpositive baseline occupied/virtual gap in spin {spin}; "
                "the fixed-reference response is not well-defined."
            )

    density_fit = getattr(mf, "with_df", None)
    data = {
        "zmp_constraint_version": 1,
        "zmp_constraint_mol": mol.dumps(),
        "zmp_constraint_overlap": overlap,
        "zmp_constraint_dm": densities,
        "zmp_constraint_fock": focks,
        "zmp_constraint_mo_coeff": coeffs,
        "zmp_constraint_mo_occ": occs,
        "zmp_constraint_mo_energy": energies,
        "zmp_constraint_coords": grids.coords,
        "zmp_constraint_weights": grids.weights,
        "zmp_constraint_grid_level": grids.level,
        "zmp_constraint_cube_type": grids.cube_type,
        "zmp_constraint_cube_size": grids.cube_size,
        "zmp_constraint_edge_len": EDGE_LEN,
        "zmp_constraint_input_level": grids.input_level,
        "zmp_constraint_baseline_xc": mf.xc,
        "zmp_constraint_baseline_density_fit": density_fit is not None,
        "zmp_constraint_baseline_auxmol": (
            density_fit.auxmol.dumps() if density_fit is not None else ""
        ),
        "zmp_constraint_baseline_grid_coords": mf.grids.coords,
        "zmp_constraint_baseline_grid_weights": mf.grids.weights,
        "zmp_constraint_jacobian_convention": "dfeature/dkappa_ai; dP=n*(Cv*kappa*Co.T+h.c.)",
    }
    if response:
        operators, labels, nuclear = _observable_operators(
            mol, grids, mf._numint, block_size
        )
        total_ref = densities.sum(axis=0)
        total_target = dm_target if restricted else dm_target.sum(axis=0)
        data.update({
            "zmp_observable_labels": labels,
            "zmp_observable_operators": operators,
            "zmp_observable_nuclear": nuclear,
            "zmp_observable_reference": np.einsum("kpq,qp->k", operators, total_ref) + nuclear,
            "zmp_observable_target": np.einsum("kpq,qp->k", operators, total_target) + nuclear,
            "zmp_response_gap_floor": gap_floor,
            "zmp_response_projection_rcond": 1e-10,
            "zmp_response_model": "independent_particle; block_diagonal_baseline_fock; no_induced_Hxc",
            "zmp_response_origin_bohr": np.zeros(3, dtype=np.float64),
            "zmp_response_gaussian_width_bohr": 1.0,
        })

    prefix = Path(cache_prefix)
    pending = []
    for spin, (c, n, f, e) in enumerate(zip(coeffs, occs, focks, energies)):
        occupied = n > 0
        co, cv = c[:, occupied], c[:, ~occupied]
        occupation = 2.0 if restricted else 1.0
        gaps = e[~occupied, None] - e[None, occupied]
        fock_ov = cv.T @ f @ co
        data[f"zmp_fock_ov_s{spin}"] = fock_ov
        data[f"zmp_gap_s{spin}"] = gaps
        maps = {}
        shape = (len(grids.coords), grids.input_level, grids.cube_size)
        if response:
            potential_ov = np.einsum("pa,kpq,qi->kai", cv, operators, co, optimize=True)
            projection = np.linalg.pinv(
                potential_ov.reshape(len(operators), -1).T, rcond=1e-10
            )
            data[f"zmp_potential_ov_s{spin}"] = potential_ov
            data[f"zmp_baseline_potential_coeff_s{spin}"] = projection @ fock_ov.ravel()
            data[f"zmp_observable_response_s{spin}"] = -2 * occupation * np.einsum(
                "mai,kai,ai->mk", potential_ov, potential_ov,
                1 / np.maximum(gaps, gap_floor), optimize=True,
            )
            path = prefix.with_name(prefix.name + f"_potential_feature_s{spin}.npy")
            maps["potential"] = np.lib.format.open_memmap(
                str(path) + ".partial", mode="w+", dtype=np.float64,
                shape=shape + (len(operators),),
            )
            pending.append((maps["potential"], path))
            data[f"zmp_potential_feature_file_s{spin}"] = path.name
        if feature_ov:
            path = prefix.with_name(prefix.name + f"_feature_ov_s{spin}.npy")
            maps["ov"] = np.lib.format.open_memmap(
                str(path) + ".partial", mode="w+", dtype=np.float64,
                shape=shape + gaps.shape,
            )
            pending.append((maps["ov"], path))
            data[f"zmp_feature_ov_file_s{spin}"] = path.name
        for start in range(0, len(grids.coords), block_size):
            stop = min(start + block_size, len(grids.coords))
            cube = grids.gen_cube(mol, dm_zmp, grids.coords[start:stop])
            generate = cube.gen_cube_rho_rks if restricted else cube.gen_cube_rho_uks
            _, chain, ao = generate(mf._numint, dm_zmp, ao_deriv=1)
            chain = chain.reshape(
                (grids.input_level, 4, -1) if restricted
                else (grids.input_level, 2, 4, -1)
            )
            if not restricted:
                chain = chain[:, spin]
            ao_occ = ao[:4] @ co
            ao_vir = ao[:4] @ cv
            # The gradient channel differentiates both AO factors; the value
            # channel is already included by the first contraction.
            jac = np.einsum(
                "fdg,dga,gi->fgai", chain, ao_vir, ao_occ[0], optimize=True
            )
            jac += np.einsum(
                "fdg,ga,dgi->fgai", chain[:, 1:], ao_vir[0], ao_occ[1:],
                optimize=True,
            )
            jac *= 2 * occupation
            jac = jac.reshape(
                grids.input_level, stop - start, grids.cube_size, *gaps.shape
            ).transpose(1, 0, 2, 3, 4)
            if feature_ov:
                maps["ov"][start:stop] = jac
            if response:
                maps["potential"][start:stop] = np.einsum(
                    "ka,cfqa->cfqk", projection,
                    jac.reshape(*jac.shape[:3], -1), optimize=True,
                ) / (2 * occupation)
    for array, path in pending:
        array.flush()
        Path(str(path) + ".partial").replace(path)
    return data
