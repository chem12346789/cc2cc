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
    mf.max_cycle = 200
    mf.kernel()
    if mf.converged is False:
        raise ValueError("RHF not converged.")
    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel(mf.make_rdm1())
    if mdft.converged is False:
        raise ValueError("RKS not converged.")

    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]

    if TEST:
        dm1_cc = mdft.make_rdm1(ao_repr=True)
        e_cc = mdft.e_tot
        rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        rho_cube = grids.gen_cube_rho(mol, dm1_cc)

        exc_cc_grids = pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0]
        h1e = mdft.mol.intor("int1e_kin") + mdft.mol.intor("int1e_nuc")
        eri = mdft.mol.intor("int2e")
        error_energy = e_cc - (
            np.einsum("pq,pq", h1e, dm1_cc)
            + 0.5 * np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc)
            - 0.05 * np.einsum("pqrs,pr,qs", eri, dm1_cc, dm1_cc)
            + mdft.energy_nuc()
        )
        print(
            f"Error energy: {AU2KCALMOL * (error_energy - np.sum(exc_cc_grids * grids.weights))}"
        )
        hybrid_exc = grids.get_center_density(rho_cube)
        print(
            AU2KCALMOL
            * np.linalg.norm(
                exc_cc_grids
                - (
                    0.08 * hybrid_exc[:, 0]
                    + 0.19 * hybrid_exc[:, 1]
                    + 0.72 * hybrid_exc[:, 2]
                    + 0.81 * hybrid_exc[:, 3]
                )
            )
        )
    else:
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()
        if mycc.converged is False:
            raise ValueError("CCSD not converged.")
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        e_dft = mdft.energy_tot(dm1_dft)

        rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        rho_cube = grids.gen_cube_rho(mol, dm1_cc)
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

        dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
        eigs_v_dm1 = mf.mo_coeff @ eigs_v_dm1
        for i in range(np.shape(eigs_v_dm1)[1]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                eigs_v_dm1[:, i],
                eigs_v_dm1[:, i],
                ao_2_diag,
            )
            exc_cc_grids -= part * eigs_e_dm1[i] / 2

        for i in range(mol.nelec[0]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                mdft.mo_coeff[:, i],
                mdft.mo_coeff[:, i],
                ao_2_diag,
            )
            exc_cc_grids += part

        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-3:
                    exc_cc_grids[i] -= (
                        (rho_cc[0][i] - rho_dft[0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
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

    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        rho_cube=rho_cube,
        weights=grids.weights,
        exc_cc_grids=exc_cc_grids,
        error_energy=error_energy,
    )
