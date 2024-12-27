import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
import opt_einsum as oe

from cc2cc.utils import Grid
from cc2cc.utils import DATA_PATH, AU2KCALMOL, TEST


def ucc(mol, name):
    """
    Generate data for the UCCSD method.
    """

    print(f"Generate data for {name}, spin {mol.spin}")

    mdft = pyscf.scf.UKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    grids = Grid(mol)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]

    if TEST:
        dm1_cc = mdft.make_rdm1(ao_repr=True)
        e_cc = mdft.e_tot
        rho_dft = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA"),
        ]
        rho_cube = grids.gen_cube_rho(mol, dm1_cc)

        exc_cc_grids = pyscf.dft.libxc.eval_xc(
            "b3lyp", rho_dft, spin=1 if mol.spin else 0
        )[0] * (rho_dft[0][0] + rho_dft[1][0])
        h1e = mdft.mol.intor("int1e_kin") + mdft.mol.intor("int1e_nuc")
        eri = mdft.mol.intor("int2e")
        error_energy = e_cc - (
            np.einsum("pq,pq", h1e, dm1_cc[0] + dm1_cc[1])
            + 0.5
            * np.einsum("pqrs,pq,rs", eri, dm1_cc[0] + dm1_cc[1], dm1_cc[0] + dm1_cc[1])
            - 0.1 * np.einsum("pqrs,pr,qs", eri, dm1_cc[0], dm1_cc[0])
            - 0.1 * np.einsum("pqrs,pr,qs", eri, dm1_cc[1], dm1_cc[1])
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
        mf = pyscf.scf.UHF(mol)
        mf.kernel()
        mycc = pyscf.cc.UCCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        e_dft = mdft.energy_tot(dm1_dft)

        rho_dft = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
        ]
        rho_cc = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA"),
        ]
        rho_cube = grids.gen_cube_rho(mol, dm1_dft)

        dm12 = (
            0.5 * dm2_cc[0]
            + 0.5 * dm2_cc[1]
            + 0.5 * dm2_cc[1].transpose(2, 3, 0, 1)
            + 0.5 * dm2_cc[2]
            - 0.5
            * oe.contract(
                "pq,rs->pqrs",
                dm1_dft[0] + dm1_dft[1],
                dm1_dft[0] + dm1_dft[1],
            )
            + 0.1
            * oe.contract(
                "pr,qs->pqrs",
                dm1_dft[0],
                dm1_dft[0],
            )
            + 0.1
            * oe.contract(
                "pr,qs->pqrs",
                dm1_dft[1],
                dm1_dft[1],
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
            "b3lyp", rho_dft, spin=1 if mol.spin else 0
        )[0] * (rho_dft[0][0] + rho_dft[1][0])

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
        for i_spin in range(2):
            eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo[i_spin])
            eigs_v_dm1 = mf.mo_coeff[i_spin] @ eigs_v_dm1
            for i in range(np.shape(eigs_v_dm1)[1]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_value[0],
                    eigs_v_dm1[:, i],
                    eigs_v_dm1[:, i],
                    ao_2_diag,
                )
                exc_cc_grids -= part * eigs_e_dm1[i] / 2

            for i in range(mol.nelec[i_spin]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_value[0],
                    mdft.mo_coeff[i_spin][:, i],
                    mdft.mo_coeff[i_spin][:, i],
                    ao_2_diag,
                )
                exc_cc_grids += part / 2

        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-3:
                    exc_cc_grids[i] -= (
                        (
                            rho_cc[0][0][i]
                            + rho_cc[1][0][i]
                            - rho_dft[0][0][i]
                            - rho_dft[1][0][i]
                        )
                        * mol.atom_charges()[i_atom]
                        / distance
                    )

        nuc = mol.intor("int1e_nuc")
        error_nuc = np.einsum("pq,pq", nuc, dm1_cc[0] + dm1_cc[1]) - np.einsum(
            "pq,pq", nuc, dm1_dft[0] + dm1_dft[1]
        )
        kin = mol.intor("int1e_kin")
        error_kin = np.einsum("pq,pq", kin, dm1_cc[0] + dm1_cc[1]) - np.einsum(
            "pq,pq", kin, dm1_dft[0] + dm1_dft[1]
        )
        eri = mol.intor("int2e")
        error_eris = 0.5 * (
            np.einsum("pqrs,pq,rs", eri, dm1_cc[0] + dm1_cc[1], dm1_cc[0] + dm1_cc[1])
            - np.einsum(
                "pqrs,pq,rs", eri, dm1_dft[0] + dm1_dft[1], dm1_dft[0] + dm1_dft[1]
            )
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
