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
    gradient = grad.kernel()
    grid_force, vnlc_deriv = rks_grad.get_nlc_vxc_full_response(
        mf._numint, mol, nlcgrids, "vv10", dm_tot, max_memory=mf.max_memory
    )
    gradient += grid_force
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        gradient[atom] += 2 * np.einsum(
            "xij,ij->x", vnlc_deriv[:, p0:p1], dm_tot[p0:p1]
        )

    return gradient
