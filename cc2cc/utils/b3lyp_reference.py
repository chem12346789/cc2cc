"""Shared B3LYP/VV10 data generation and finite-difference references."""

from __future__ import annotations

import numpy as np

import pyscf.dft

from cc2cc.utils.Grids import Grid
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.TestDataDFT import diff_rho
from cc2cc.utils.nlc import eval_nlc_exc_density, post_dft_vv10_gradient


def get_vv10_data(
    mol,
    grids,
    dm_dft,
    e_dft,
    dm_cc,
    args,
    suffix="",
):
    """Compute VV10 data using the calculation policy selected by suffix.

    Energies are in Hartree and nuclear gradients in Hartree/Bohr.
    The "_zmp" suffix keeps post-DFT energies and gradients but skips SCF-VV10,
    finite-difference checks, and density comparisons. Other suffixes enable all.
    """
    evaluate = suffix == "_zmp"
    calculate_scf_vv10 = not evaluate
    data = get_post_vv10_data(
        mol, grids, dm_dft, e_dft, dm_cc, args, evaluate,
        check_post_vv10_gradient=not evaluate,
    )
    if calculate_scf_vv10:
        data.update(get_scf_vv10_data(mol, grids, dm_dft, dm_cc, args, evaluate))
    print("Energy relative to B3LYP total (kcal/mol):")
    print("  B3LYP baseline:                      0.0")
    print(
        "  Post-DFT VV10:                      "
        f"{(data['e_post_vv10'] - e_dft) * AU2KCALMOL:>16.8f}"
    )
    if calculate_scf_vv10:
        print(
            "  Self-consistent SCF-VV10:            "
            f"{(data['e_scf_vv10'] - e_dft) * AU2KCALMOL:>16.8f}"
        )
        print(
            "  SCF-VV10 minus post-DFT:             "
            f"{(data['e_scf_vv10'] - data['e_post_vv10']) * AU2KCALMOL:>16.8f}"
        )
    if suffix:
        data = {f"{key}{suffix}": value for key, value in data.items()}
    return data


def get_post_vv10_data(
    mol, grids, dm_dft, e_dft, dm_cc, args, evaluate=False,
    check_post_vv10_gradient=False,
):
    """Compute post-DFT VV10 on the supplied RKS or UKS density."""
    dm_tot = dm_dft if mol.spin == 0 else dm_dft[0] + dm_dft[1]
    exc_post_vv10_grid = eval_nlc_exc_density(mol, grids, dm_tot, "vv10")
    enlc = np.dot(exc_post_vv10_grid, grids.weights)
    _, enlc_check, _ = pyscf.dft.numint.NumInt().nr_nlc_vxc(mol, grids, "vv10", dm_tot)
    print(
        "NLC energy check (kcal/mol): difference = "
        f"{(enlc - enlc_check) * AU2KCALMOL:.12e}"
    )
    if not np.isclose(enlc, enlc_check, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "NLC grid integral disagrees with nr_nlc_vxc: "
            f"{enlc} vs {enlc_check} Hartree"
        )
    data = {
        "exc_post_vv10_grid": exc_post_vv10_grid,
        "e_post_vv10": e_dft + enlc,
    }
    if mol.natm == 1:
        data["grad_post_vv10"] = np.zeros((mol.natm, 3))
    else:
        mf_dft = converged_b3lyp(mol, dm0=dm_dft)
        if not mf_dft.converged:
            raise RuntimeError("Post-DFT VV10 B3LYP reference did not converge.")
        data["grad_post_vv10"] = post_dft_vv10_gradient(mf_dft, grids)
        if not evaluate and check_post_vv10_gradient:
            data["grad_post_vv10_fd"] = post_vv10_finite_difference_gradient(
                mol, args.grid_level
            )
            grad_error = np.max(
                np.abs(data["grad_post_vv10"] - data["grad_post_vv10_fd"])
            )
            print(
                "Post-DFT VV10 analytical/FD gradient max difference "
                f"(Hartree/Bohr): {grad_error:.12e}"
            )
            grad_dft = mf_dft.nuc_grad_method()
            grad_dft.grid_response = True
            grad_dft_error = np.max(
                np.abs(grad_dft.kernel() - b3lyp_finite_difference_gradient(mol))
            )
            print(
                "DFT (B3LYP) analytical/FD gradient max difference "
                f"(Hartree/Bohr): {grad_dft_error:.12e}"
            )
    if not evaluate:
        print(
            "Integrated absolute density difference "
            "(B3LYP/post-DFT VV10 vs CCSD, electrons): "
            f"{diff_rho(mol, dm_dft, dm_cc, grids):.10e}"
        )
    return data


def get_scf_vv10_data(mol, grids, dm_dft, dm_cc, args, evaluate=False):
    """Compute self-consistent B3LYP+VV10 data starting from an RKS or UKS density."""
    mf_scf = converged_b3lyp(mol, dm0=dm_dft, nlc="vv10", verbose=4)
    if not mf_scf.converged and (args.check_convergence or not evaluate):
        mf_scf.level_shift = 1
        mf_scf.kernel(dm0=dm_dft)
        if not mf_scf.converged:
            raise RuntimeError("SCF-VV10 did not converge.")
    dm_scf_vv10 = mf_scf.make_rdm1()
    data = {
        "e_scf_vv10": float(mf_scf.e_tot),
        "dm1_scf_vv10": dm_scf_vv10,
    }
    if not evaluate:
        if mol.natm == 1:
            data["grad_scf_vv10"] = np.zeros((mol.natm, 3))
        else:
            grad_scf = mf_scf.nuc_grad_method()
            grad_scf.grid_response = True
            data["grad_scf_vv10"] = grad_scf.kernel()
        for label, left, right in (
            ("B3LYP vs SCF-VV10", dm_dft, dm_scf_vv10),
            ("SCF-VV10 vs CCSD", dm_scf_vv10, dm_cc),
        ):
            print(
                f"Integrated absolute density difference ({label}, electrons): "
                f"{diff_rho(mol, left, right, grids):.10e}"
            )
    return data


def converged_b3lyp(mol, dm0=None, *, nlc="", grids=None, verbose=0, use_gpu=False):
    mf = (
        pyscf.dft.RKS(mol, xc="b3lyp")
        if mol.spin == 0
        else pyscf.dft.UKS(mol, xc="b3lyp")
    )
    if grids is not None:
        mf.grids = grids
    mf.verbose = verbose
    mf.max_cycle = 200
    mf.nlc = nlc
    if use_gpu:
        mf = mf.to_gpu()
    mf.kernel(dm0=dm0)
    return mf


def post_vv10_total_energy(mol, grid_level):
    mf = converged_b3lyp(mol)
    dm = mf.make_rdm1()
    dm_tot = dm if mol.spin == 0 else dm[0] + dm[1]
    grids = Grid(mol, grid_level, 7)
    enlc = np.dot(
        eval_nlc_exc_density(mol, grids, dm_tot, "vv10"),
        np.asarray(grids.weights),
    )
    return float(mf.e_tot) + enlc


def post_vv10_finite_difference_gradient(mol, grid_level, step=1e-3):
    """Central-difference reference gradient in Hartree/Bohr; step in Bohr."""
    return finite_difference_gradient(
        mol, lambda displaced: post_vv10_total_energy(displaced, grid_level), step
    )


def b3lyp_finite_difference_gradient(mol, step=1e-3):
    """Central-difference B3LYP gradient in Hartree/Bohr; step in Bohr."""
    return finite_difference_gradient(
        mol, lambda displaced: converged_b3lyp(displaced).e_tot, step
    )


def finite_difference_gradient(mol, total_energy, step):
    coords = mol.atom_coords()
    gradient = np.empty_like(coords)
    for atom in range(mol.natm):
        for axis in range(3):
            displaced_energies = []
            for sign in (1, -1):
                displaced_mol = mol.copy()
                displaced_coords = coords.copy()
                displaced_coords[atom, axis] += sign * step
                displaced_mol.set_geom_(displaced_coords, unit="Bohr")
                displaced_energies.append(total_energy(displaced_mol))
            gradient[atom, axis] = (displaced_energies[0] - displaced_energies[1]) / (
                2 * step
            )
    return gradient
