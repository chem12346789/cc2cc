# pylint: disable=W0212

import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
import opt_einsum as oe

from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
from pyscf.cc import uccsd_t
from pyscf.cc import uccsd_rdm
from pyscf.cc.uccsd_t_rdm import _gamma1_intermediates as u_gamma1_intermediates
from pyscf.cc.uccsd_t_rdm import _gamma2_intermediates as u_gamma2_intermediates

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL


def get_dft_energy(
    mol,
    grids,
    mf_mo_coeff,
    dm1_dft,
    dft_mo_coeff,
    e_dft,
    dm1_cc,
    dm1_cc_mo,
    dm2_cc,
    e_cc,
):
    """
    Calculate the (exchange-correlation energy - DFT energy) on the grids.
    """
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]

    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
    ]
    rho_cc = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA"),
    ]

    dm12 = (
        0.5 * dm2_cc[0]
        + 0.5 * dm2_cc[1]
        + 0.5 * dm2_cc[1].transpose(2, 3, 0, 1)
        + 0.5 * dm2_cc[2]
        - 0.5
        * oe.contract("pq,rs->pqrs", dm1_dft[0] + dm1_dft[1], dm1_dft[0] + dm1_dft[1])
        + 0.1 * oe.contract("pr,qs->pqrs", dm1_dft[0], dm1_dft[0])
        + 0.1 * oe.contract("pr,qs->pqrs", dm1_dft[1], dm1_dft[1])
    )
    # exchange part
    # + 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_cc[0], dm1_cc[0])
    # + 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_cc[1], dm1_cc[1])
    # alpha is 0.2 in b3lyp

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

    for i_spin in range(2):
        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo[i_spin])
        eigs_v_dm1 = mf_mo_coeff[i_spin] @ eigs_v_dm1
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
                dft_mo_coeff[i_spin][:, i],
                dft_mo_coeff[i_spin][:, i],
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

    error_energy = e_cc - e_dft
    error = np.sum(exc_cc_grids * grids.weights) - error_energy
    print(
        "exc_cc_grids: ",
        f"max exc_cc_grids: {np.max(exc_cc_grids)}",
        f"min exc_cc_grids: {np.min(exc_cc_grids)}",
        f"mean exc_cc_grids: {np.mean(exc_cc_grids)}",
        f"std exc_cc_grids: {np.std(exc_cc_grids)}",
        f"error_energy: {AU2KCALMOL * error_energy},",
        f"Error: {AU2KCALMOL * error},",
    )
    return error_energy, exc_cc_grids, rho_cc, rho_dft


def ucc(mol, grids, name, args):
    """
    Generate data for the UCCSD method.
    """
    print(f"Generate data for {name}, spin {mol.spin}")

    mf = pyscf.scf.UHF(mol).newton()
    mf.max_cycle = 200
    mf.kernel()
    if args.check_convergence and not mf.converged:
        raise ValueError("UHF not converged.")

    mdft = pyscf.scf.UKS(mol)
    mdft.verbose = 4
    mdft.max_cycle = 200
    mdft.xc = "b3lyp"
    mdft.kernel(mf.make_rdm1())
    if args.check_convergence and not mdft.converged:
        raise ValueError("UKS not converged.")
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    e_dft = mdft.energy_tot(dm1_dft)

    mycc = pyscf.cc.UCCSD(mf)
    mycc.verbose = 4
    _, t1, t2 = mycc.kernel()
    if args.cc_triple:
        eris = mycc.ao2mo()
        e3ref = uccsd_t.kernel(mycc, eris, t1, t2)
        l1, l2 = uccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
        dm1_cc = uccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris, ao_repr=True)
        dm1_cc_mo = uccsd_t_rdm.make_rdm1(
            mycc, t1, t2, l1, l2, eris=eris, ao_repr=False
        )
        d1 = u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        d2 = u_gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
        dm2_cc = uccsd_rdm._make_rdm2(mycc, d1, d2, True, True, ao_repr=True)
        del d1, d2
        e_cc = mycc.e_tot + e3ref
        print(f"UCCSD(T) energy: {e3ref}")
    else:
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot
    dm1_cc = np.array(dm1_cc)
    dm2_cc = np.array(dm2_cc)

    print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} (CCSD vs DFT)")
    cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
    dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
    print(f"{np.linalg.norm(cc_dipole - dft_dipole)} (CCSD vs DFT)")

    error_energy_dft, exc_cc_grids_dft, rho_cc, rho_dft = get_dft_energy(
        mol,
        grids,
        mf.mo_coeff,
        dm1_dft,
        mdft.mo_coeff,
        e_dft,
        dm1_cc,
        dm1_cc_mo,
        dm2_cc,
        e_cc,
    )

    rho_cube_cc = grids.gen_cube_rho_uks(rho_cc, mdft._numint, dm1_cc)
    rho_cube_dft = grids.gen_cube_rho_uks(rho_dft, mdft._numint, dm1_dft)
    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm1_cc=dm1_cc,
        rho_cube_cc=rho_cube_cc,
        rho_cube_dft=rho_cube_dft,
        weights=grids.weights,
        exc_cc_grids=exc_cc_grids_dft,
        error_energy=error_energy_dft,
        mol=mol.tostring(format="xyz"),
        charge=mol.charge,
        spin=mol.spin,
    )
