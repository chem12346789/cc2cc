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
    dm1_dft = mdft.make_rdm1(ao_repr=True)

    grids = Grid(mol)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    # ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    # rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
    dm1_input = dm1_cc.copy()
    rho_input = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_input, xctype="GGA")
    exc_output_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_input)[0] * rho_input[0]

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
            exc_output_grids[i] += expr_rinv_dm2_r(
                ao_0_i,
                ao_0_i,
                rinv,
                backend="torch",
            )

    #     for i_atom in range(mol.natm):
    #         exc_cc_grids[i] -= (
    #             (rho_cc[0][i] - rho_dft[0][i])
    #             * mol.atom_charges()[i_atom]
    #             / (np.linalg.norm(mol.atom_coords()[i_atom] - coord))
    #         )

    # dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
    # eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
    # eigs_v_dm1 = mf.mo_coeff @ eigs_v_dm1
    # for i in range(np.shape(eigs_v_dm1)[1]):
    #     part = oe.contract(
    #         "pm,m,n,pn->p",
    #         ao_value[0],
    #         eigs_v_dm1[:, i],
    #         eigs_v_dm1[:, i],
    #         ao_2_diag,
    #     )
    #     exc_cc_grids -= part * eigs_e_dm1[i] / 2

    # for i in range(mol.nelec[0]):
    #     part = oe.contract(
    #         "pm,m,n,pn->p",
    #         ao_value[0],
    #         mdft.mo_coeff[:, i],
    #         mdft.mo_coeff[:, i],
    #         ao_2_diag,
    #     )
    #     exc_cc_grids += part

    error_energy = e_cc - mdft.energy_tot(dm1_input)
    error = np.sum(exc_output_grids * grids.weights) - error_energy
    print(
        "exc_output_grids: ",
        f"max exc_output_grids: {np.max(exc_output_grids)}",
        f"min exc_output_grids: {np.min(exc_output_grids)}",
        f"mean exc_output_grids: {np.mean(exc_output_grids)}",
        f"var exc_output_grids: {np.var(exc_output_grids)}",
    )
    print(
        f"error_energy: {AU2KCALMOL * error_energy},",
        f"Error: {AU2KCALMOL * error},",
    )

    rho_cube = grids.gen_cube_rho(mol, dm1_dft)

    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        rho_cube=rho_cube,
        weights=grids.weights,
        exc_output_grids=exc_output_grids,
        exc_over_rho_output_grids=exc_output_grids / (rho_input + 1e-14),
        error_energy=error_energy,
    )
