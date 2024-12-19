import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
import opt_einsum as oe

from cc2cc.utils import Grid
from cc2cc.utils import DATA_PATH, AU2KCALMOL

TEST = True


def ucc(mol, name):
    """
    Generate data for the UCCSD method.
    """

    print(f"Generate data for {name}, spin {mol.spin}")

    mdft = pyscf.scf.UKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    if TEST:
        dm1_cc = mdft.make_rdm1(ao_repr=True)
        e_cc = mdft.e_tot
    else:
        mf = pyscf.scf.UHF(mol)
        mf.kernel()
        mycc = pyscf.cc.UCCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot

    grids = Grid(mol)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
    rho_cc = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA"),
    ]

    if TEST:
        exc_cc_grids = pyscf.dft.libxc.eval_xc(
            "b3lyp", rho_cc, spin=1 if mol.spin else 0
        )[0] * (rho_cc[0][0] + rho_cc[1][0])
        h1e = mdft.mol.intor("int1e_kin") + mdft.mol.intor("int1e_nuc")
        eri = mdft.mol.intor("int2e")
        error_energy = (
            np.einsum("pq,pq", h1e, dm1_cc[0] + dm1_cc[1])
            + 0.5
            * np.einsum("pqrs,pq,rs", eri, dm1_cc[0] + dm1_cc[1], dm1_cc[0] + dm1_cc[1])
            - 0.1 * np.einsum("pqrs,pr,qs", eri, dm1_cc[0], dm1_cc[0])
            - 0.1 * np.einsum("pqrs,pr,qs", eri, dm1_cc[1], dm1_cc[1])
            + mdft.energy_nuc()
            + np.sum(exc_cc_grids * grids.weights)
            - e_cc
        )
        print(f"Error energy: {AU2KCALMOL * error_energy}")
    else:
        dm12 = (
            0.5 * dm2_cc[0]
            + 0.5 * dm2_cc[1]
            + 0.5 * dm2_cc[1].transpose(2, 3, 0, 1)
            + 0.5 * dm2_cc[2]
            - 0.5
            * oe.contract(
                "pq,rs->pqrs",
                dm1_cc[0] + dm1_cc[1],
                dm1_cc[0] + dm1_cc[1],
            )
            + 0.1
            * oe.contract(
                "pr,qs->pqrs",
                dm1_cc[0],
                dm1_cc[0],
            )
            + 0.1
            * oe.contract(
                "pr,qs->pqrs",
                dm1_cc[1],
                dm1_cc[1],
            )
        )

        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            dm12,
            (mol.nao,),
            (mol.nao,),
            (mol.nao, mol.nao),
            constants=[0],
            optimize="optimal",
        )

        exc_cc_grids = -pyscf.dft.libxc.eval_xc(
            "b3lyp", rho_cc, spin=1 if mol.spin else 0
        )[0] * (rho_cc[0][0] + rho_cc[1][0])

        for i, coord in enumerate(grids.coords):
            if i * 10 % len(grids.coords) == 0:
                print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)

            with mol.with_rinv_origin(coord):
                rinv = mol.intor("int1e_rinv")
                exc_cc_grids[i] += expr_rinv_dm2_r(
                    ao_value[0][i],
                    ao_value[0][i],
                    rinv,
                    backend="torch",
                )

        error_energy = e_cc - mdft.energy_tot(dm1_cc)
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"max exc_cc_grids: {np.max(exc_cc_grids)}",
            f"min exc_cc_grids: {np.min(exc_cc_grids)}",
            f"mean exc_cc_grids: {np.mean(exc_cc_grids)}",
            f"var exc_cc_grids: {np.var(exc_cc_grids)}",
        )
        print(
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

    rho_cube = grids.gen_cube_rho(mol, dm1_cc)

    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        rho_cube=rho_cube,
        weights=grids.weights,
        exc_cc_grids=exc_cc_grids,
        exc_dft_grids=exc_dft_grids,
        error_energy=error_energy,
    )
