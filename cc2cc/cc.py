import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
import opt_einsum as oe

from cc2cc.utils import Grid
from cc2cc.utils import DATA_PATH, AU2KCALMOL


def cc(mol, name):
    """
    Generate data for the CCSD method. (Restrict scenario to spin 0).
    """

    print(f"Generate data for {name}")

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()
    dm1_cc = mycc.make_rdm1(ao_repr=True)
    dm2_cc = mycc.make_rdm2(ao_repr=True)
    e_cc = mycc.e_tot

    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    grids = Grid(mol)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
    exc_over_rho_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0]

    expr_rinv_dm2_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.5 * dm2_cc
        - 0.5 * oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)
        + 0.05 * oe.contract("pr,qs->pqrs", dm1_cc, dm1_cc),
        (mol.nao,),
        (mol.nao,),
        (mol.nao, mol.nao),
        constants=[0],
        optimize="optimal",
    )

    for i, coord in enumerate(grids.coords):
        if i * 10 % len(grids.coords) == 0:
            print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)

        ao_0_i = ao_value[0][i]

        with mol.with_rinv_origin(coord):
            rinv = mol.intor("int1e_rinv")
            exc_over_rho_cc_grids[i] += expr_rinv_dm2_r(
                ao_0_i,
                ao_0_i,
                rinv,
                backend="torch",
            ) / (rho_cc[0][i] + 1e-14)

    error_energy = e_cc - mdft.energy_tot(dm1_cc)
    error = np.sum(exc_over_rho_cc_grids * rho_cc[0] * grids.weights) - error_energy
    print(
        f"error_energy: {AU2KCALMOL * error_energy},",
        f"Error: {AU2KCALMOL * error},",
    )

    rho_cube = grids.gen_cube_rho(mol, dm1_cc)

    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        weights=grids.weights,
        exc_over_dm_cc_grids=exc_over_rho_cc_grids,
        rho_cube=rho_cube,
        error_energy=error_energy,
    )
