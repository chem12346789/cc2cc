from __future__ import annotations

import numpy as np
import pyscf
import pyscf.dft


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
