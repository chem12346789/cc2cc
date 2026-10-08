from __future__ import annotations

import numpy as np
import pyscf
import pyscf.dft
import pyscf.hessian.rks
import pyscf.hessian.uks
from pyscf.grad import rks as rks_grad


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
                ao_dm = np.einsum("cgi,ij->cgj", ao[:4], density)
                derivative = 2 * np.einsum("xcgi,gi->xcg", dao, ao_dm[0])
                derivative[:, 1:] += 2 * np.einsum(
                    "cgi,xgi->xcg", ao[1:4], ao_dm[1:4]
                )
                drho.append(derivative)
            dvxc = np.einsum("scudg,uxdg->sxcg", fxc, np.array(drho))
            weight = weights[start:stop]
            for spin in range(len(dms)):
                weighted_vxc = vxc[spin] * factors[:, None]
                aow = np.einsum("cg,cgi->gi", weighted_vxc, ao[:4])
                response[spin] += np.einsum(
                    "axg,gi,gj->axij", weight1[:, :, start:stop],
                    ao[0], aow, optimize=True,
                )
                daow = np.einsum(
                    "xcg,cgi->xgi", dvxc[spin] * factors[None, :, None], ao[:4]
                )
                daow += np.einsum("cg,xcgi->xgi", weighted_vxc, dao)
                response[spin, owner] += np.einsum(
                    "xgi,gj->xij", dao[:, 0], aow * weight[:, None]
                )
                response[spin, owner] += np.einsum(
                    "gi,xgj->xij", ao[0], daow * weight[None, :, None]
                )
    response += response.swapaxes(-1, -2).copy()
    return response[0] if dm.ndim == 2 else response


def post_dft_vv10_gradient(mf, nlcgrids):
    """Analytical B3LYP+VV10 nuclear gradient, in Hartree/Bohr.

    ``mf`` must contain converged B3LYP RKS/UKS orbitals with NLC disabled.
    Both the B3LYP orbital response and moving integration grids are included.
    """
    if not mf.converged:
        raise RuntimeError("Post-DFT VV10 gradient requires converged B3LYP.")
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
        gradient[atom] += 2 * np.einsum(
            "xij,ij->x", vnlc_deriv[:, p0:p1], dm_tot[p0:p1]
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
            gradient[atom] += 2 * np.einsum(
                "ij,xjk,ik->x", vnlc, response[atom], occupied
            )
    return gradient
