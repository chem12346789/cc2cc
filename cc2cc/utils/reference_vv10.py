"""Shared VV10 data generation and finite-difference references."""

from __future__ import annotations

import numpy as np

import pyscf
import pyscf.dft
from pyscf.grad import rks as rks_grad

from cc2cc.utils.Grids import Grid
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.TestDataDFT import diff_rho
from cc2cc.utils.reference_b3lyp import (
    b3lyp_finite_difference_gradient,
    converged_b3lyp,
    finite_difference_gradient,
)
from cc2cc.utils.tensor_helper import parallel_einsum


def eval_nlc_exc_density(
    mol: pyscf.M,
    grids: pyscf.dft.gen_grid.Grids,
    dm: np.ndarray,
    xc_code: str = "vv10",
    max_memory: float = 2000,
) -> np.ndarray:
    """Evaluate NLC energy density at each grid point (energy/volume)."""
    ni = pyscf.dft.numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)
    assert nset == 1

    rho_blocks = []
    for ao, mask, _, _ in ni.block_loop(
        mol, grids, nao, deriv=1, max_memory=max_memory
    ):
        rho_blocks.append(make_rho(0, ao, mask, "GGA"))
    rho = np.hstack(rho_blocks)

    exc = np.zeros_like(rho[0])
    for nlc_pars, fac in ni.nlc_coeff(xc_code):
        exc_block, _ = pyscf.dft.numint._vv10nlc(
            rho, grids.coords, rho, grids.weights, grids.coords, nlc_pars
        )
        exc += exc_block * fac
    return rho[0] * exc


def _gga_potential_grid_response(mf):
    """Grid-motion part of the GGA Fock derivative omitted by make_h1."""
    mol = mf.mol
    ni = mf._numint
    dm = mf.make_rdm1()
    dms = dm[None] if dm.ndim == 2 else dm
    nao = mol.nao_nr()
    response = np.zeros((len(dms), mol.natm, 3, nao, nao), dtype=np.float64)
    second_derivatives = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    factors = np.array([0.5, 1.0, 1.0, 1.0], dtype=np.float64)
    for owner, (coords, weights, weight1) in enumerate(
        rks_grad.grids_response_cc(mf.grids)
    ):
        for start in range(0, len(weights), pyscf.dft.numint.BLKSIZE * 8):
            stop = start + pyscf.dft.numint.BLKSIZE * 8
            ao = ni.eval_ao(mol, coords[start:stop], deriv=2)
            dao = np.stack([
                ao[[axis + 1, *second_derivatives[axis]]] for axis in range(3)
            ])
            rho = np.array([
                ni.eval_rho(mol, ao[:4], density, xctype="GGA", hermi=1)
                for density in dms
            ])
            xc_rho = rho[0] if dm.ndim == 2 else rho
            _, vxc, fxc, _ = ni.eval_xc_eff(mf.xc, xc_rho, deriv=2, xctype="GGA")
            if dm.ndim == 2:
                vxc = vxc[None]
                fxc = fxc[None, :, None, :, :]
            drho = []
            for density in dms:
                ao_dm = parallel_einsum("cgi,ij->cgj", ao[:4], density)
                derivative = 2 * parallel_einsum("xcgi,gi->xcg", dao, ao_dm[0])
                derivative[:, 1:] += 2 * parallel_einsum(
                    "cgi,xgi->xcg", ao[1:4], ao_dm[1:4],
                )
                drho.append(derivative)
            dvxc = parallel_einsum("scudg,uxdg->sxcg", fxc, np.array(drho))
            weight = weights[start:stop]
            for spin in range(len(dms)):
                weighted_vxc = vxc[spin] * factors[:, None]
                aow = parallel_einsum("cg,cgi->gi", weighted_vxc, ao[:4])
                response[spin] += parallel_einsum(
                    "axg,gi,gj->axij", weight1[:, :, start:stop],
                    ao[0], aow,
                )
                daow = parallel_einsum(
                    "xcg,cgi->xgi", dvxc[spin] * factors[None, :, None], ao[:4],
                )
                daow += parallel_einsum("cg,xcgi->xgi", weighted_vxc, dao)
                response[spin, owner] += parallel_einsum(
                    "xgi,gj->xij", dao[:, 0], aow * weight[:, None],
                )
                response[spin, owner] += parallel_einsum(
                    "gi,xgj->xij", ao[0], daow * weight[None, :, None],
                )
    response += response.swapaxes(-1, -2).copy()
    return response[0] if dm.ndim == 2 else response


def post_dft_vv10_gradient(mf, nlcgrids):
    """Analytical B3LYP+VV10 nuclear gradient, in Hartree/Bohr.

    ``mf`` must contain converged B3LYP RKS/UKS orbitals with NLC disabled.
    Both the B3LYP orbital response and moving integration grids are included.
    """
    if mf.do_nlc() or mf.xc.lower() != "b3lyp":
        raise ValueError("Post-DFT VV10 gradient requires B3LYP without NLC.")
    mol = mf.mol
    dm = mf.make_rdm1()
    dm_tot = dm if dm.ndim == 2 else dm.sum(axis=0)
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    gradient = grad.kernel()
    grid_force, vnlc_deriv = rks_grad.get_nlc_vxc_full_response(
        mf._numint, mol, nlcgrids, "vv10", dm_tot, max_memory=mf.max_memory
    )
    gradient += grid_force
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        gradient[atom] += 2 * parallel_einsum(
            "xij,ij->x", vnlc_deriv[:, p0:p1], dm_tot[p0:p1],
        )

    hessian = mf.Hessian()
    h1 = hessian.make_h1(mf.mo_coeff, mf.mo_occ)
    grid_h1 = _gga_potential_grid_response(mf)
    if dm.ndim == 2:
        h1 = [h1[atom] + grid_h1[atom] for atom in range(mol.natm)]
    else:
        h1 = tuple([
            h1[spin][atom] + grid_h1[spin, atom] for atom in range(mol.natm)
        ] for spin in range(2))
    mo1, _ = hessian.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1)
    _, _, vnlc = mf._numint.nr_nlc_vxc(mol, nlcgrids, "vv10", dm_tot)
    coeffs = (mf.mo_coeff,) if dm.ndim == 2 else mf.mo_coeff
    occupations = (mf.mo_occ,) if dm.ndim == 2 else mf.mo_occ
    responses = (mo1,) if dm.ndim == 2 else mo1
    for coeff, occ, response in zip(coeffs, occupations, responses):
        occupied = coeff[:, occ > 0] * occ[occ > 0]
        for atom in range(mol.natm):
            # mo1 includes the occupied-space overlap (Pulay) response.
            gradient[atom] += 2 * parallel_einsum(
                "ij,xjk,ik->x", vnlc, response[atom], occupied,
            )
    return gradient


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
