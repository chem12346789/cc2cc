import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
import opt_einsum as oe

from cc2cc.utils import DATA_PATH, AU2KCALMOL, TEST


def cc(mol, grids, name):
    """
    Generate data for the CCSD method. (Restrict scenario to spin 0).
    """

    print(f"Generate data for {name}")

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel(mf.make_rdm1())

    if TEST:
        dm1_cc = mdft.make_rdm1(ao_repr=True)
        e_cc = mdft.e_tot

        rho_norm_matrix = grids.gen_grids_matrix(mol, dm1_cc, reset=True)
        ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        print(f"Tot ele: {np.sum(rho_cc[0] * grids.weights)}")

        exc_cc_grids = pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0] * rho_cc[0]
        h1e = mdft.mol.intor("int1e_kin") + mdft.mol.intor("int1e_nuc")
        eri = mdft.mol.intor("int2e")
        error_energy = e_cc - (
            np.einsum("pq,pq", h1e, dm1_cc)
            + 0.5 * np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc)
            - 0.05 * np.einsum("pqrs,pr,qs", eri, dm1_cc, dm1_cc)
            + mdft.energy_nuc()
        )
        print(
            f"Error energy: {AU2KCALMOL * (error_energy - np.sum(exc_cc_grids * grids.weights))} KCal/Mol"
        )
    else:
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot
        dm1_dft = np.array(dm1_cc).copy()
        e_dft = mdft.energy_tot(dm1_dft)

        rho_norm_matrix = grids.gen_grids_matrix(mol, dm1_dft, reset=True)
        ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
        print(np.sum(np.abs(rho_cc - rho_dft) * grids.weights))

        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            0.5 * dm2_cc
            - 0.5 * oe.contract("pq,rs->pqrs", dm1_dft, dm1_dft)
            + 0.05 * oe.contract("pr,qs->pqrs", dm1_dft, dm1_dft),
            # exchange part
            # + 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_cc * 0.5, dm1_cc * 0.5)
            # + 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_cc * 0.5, dm1_cc * 0.5)
            # alpha is 0.2 in b3lyp
            (mol.nao,),
            (mol.nao,),
            (mol.nao, mol.nao),
            constants=[0],
            optimize="optimal",
        )

        exc_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0]

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

        error_energy = e_cc - e_dft
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"max exc_cc_grids: {np.max(exc_cc_grids)}",
            f"min exc_cc_grids: {np.min(exc_cc_grids)}",
            f"mean exc_cc_grids: {np.mean(exc_cc_grids)}",
            f"std exc_cc_grids: {np.std(exc_cc_grids)}",
        )
        print(
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

    print(exc_cc_grids - grids.matrix_to_vector(grids.vector_to_matrix(exc_cc_grids)))
    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        rho_norm_matrix=rho_norm_matrix,
        weights_matrix=grids.vector_to_matrix(grids.weights),
        exc_cc_grids_matrix=grids.vector_to_matrix(exc_cc_grids),
        error_energy=error_energy,
    )
    print(f"Save data for {name}.")
