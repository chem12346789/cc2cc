# pylint: disable=W0212

import json

import pyscf
import numpy as np
import opt_einsum as oe

from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
from pyscf.cc import uccsd_t
from pyscf.cc import uccsd_rdm
from pyscf.cc.uccsd_t_rdm import _gamma1_intermediates as u_gamma1_intermediates
from pyscf.cc.uccsd_t_rdm import _gamma2_intermediates as u_gamma2_intermediates
from pyscf.grad import uccsd_t as uccsd_t_grad

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils.modelscf_uks import get_veff_grad_modified_zeros


def get_dft_energy(
    mol,
    grids,
    dm1_dft,
    e_dft,
    mdft,
    mf,
    dm1_cc,
    dm1_cc_mo,
    dm2_cc,
    e_cc,
    evaluate=False,
):
    """
    Calculate the (exchange-correlation energy - DFT energy) on the grids.
    """
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_array = np.array([ao_value[0], ao_value[1], ao_value[2], ao_value[3]])
    ao_mat = np.array(
        [
            [ao_value[1], ao_value[2], ao_value[3]],
            [ao_value[4], ao_value[5], ao_value[6]],
            [ao_value[5], ao_value[7], ao_value[8]],
            [ao_value[6], ao_value[8], ao_value[9]],
        ]
    )
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]

    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
    ]
    rho_cc = np.zeros_like(rho_dft)  # dummy

    ni = mdft._numint
    dft_mo_coeff = mdft.mo_coeff
    mf_mo_coeff = mf.mo_coeff

    rho_input_lda = [rho_dft[0][0], rho_dft[1][0]]
    v_lda = ni.eval_xc_eff("LDA,", rho_input_lda, deriv=1, xctype="LDA")[1]
    v_vwn = ni.eval_xc_eff(",VWN3", rho_input_lda, deriv=1, xctype="LDA")[1]
    v_b88 = ni.eval_xc_eff("B88,", rho_dft, deriv=1, xctype="GGA")[1]
    v_lyp = ni.eval_xc_eff(",LYP", rho_dft, deriv=1, xctype="GGA")[1]

    vxc_b3lyp = np.zeros((4, 2, 4, len(grids.coords)))
    vxc_b3lyp[0, :, 0:1, :] = v_lda
    vxc_b3lyp[1, :, 0:1, :] = v_vwn
    vxc_b3lyp[2, :, :, :] = v_b88
    vxc_b3lyp[3, :, :, :] = v_lyp

    wv = grids.weights * vxc_b3lyp
    wv[:, :, 0, :] *= 0.5

    atmlst = range(mol.natm)
    grad2force = np.zeros((len(atmlst), 4, len(grids.coords), 3))
    for k, ia in enumerate(atmlst):
        p0, p1 = mol.aoslice_by_atom()[ia, 2:]
        grad2force[k] = np.einsum(
            "msnp,xpi,npj,sij->mpx",
            wv,
            ao_value[1:4, :, p0:p1],
            ao_array,
            dm1_dft[:, p0:p1],
            optimize=True,
        ) + np.einsum(
            "msnp,nxpi,pj,sij->mpx",
            wv,
            ao_mat[:, :, :, p0:p1],
            ao_value[0],
            dm1_dft[:, p0:p1],
            optimize=True,
        )
    grad2force = -grad2force * 2

    if evaluate:
        return None, None, rho_cc, rho_dft, grad2force
    else:
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
            * oe.contract(
                "pq,rs->pqrs", dm1_dft[0] + dm1_dft[1], dm1_dft[0] + dm1_dft[1]
            )
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
        return error_energy, exc_cc_grids, rho_cc, rho_dft, grad2force


def ucc(mol, grids, name, args, evaluate=False):
    """
    Generate data for the UCCSD method.
    """
    print(f"Generate data for {name}, spin {mol.spin}")

    # UHF calculation
    mf = pyscf.scf.UHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    if args.check_convergence and not mf.converged:
        raise ValueError("UHF not converged.")

    # UCCSD calculation
    if evaluate:
        mycc = pyscf.cc.UCCSD(mf)
        mycc.verbose = 4
        mycc.direct = True
        _, t1, t2 = mycc.kernel()
        eris = mycc.ao2mo()
        e3ref = uccsd_t.kernel(mycc, eris, t1, t2)
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm1_cc_mo = None
        dm2_cc = None
        del t1, t2
        e_cc = mycc.e_tot + e3ref
        print(f"UCCSD(T) energy: {e_cc}")
        grad_cc = np.zeros((mol.natm, 3))
    else:
        mycc = pyscf.cc.UCCSD(mf)
        mycc.verbose = 4
        _, t1, t2 = mycc.kernel()
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
        del t1, t2, l1, l2, d1, d2
        e_cc = mycc.e_tot + e3ref
        print(f"UCCSD(T) energy: {e_cc}")
        if mol.natm == 1:
            grad_cc = np.zeros((mol.natm, 3))
        else:
            gcc = uccsd_t_grad.Gradients(mycc)
            grad_cc = gcc.kernel()
    dm1_cc = np.array(dm1_cc)
    dm2_cc = np.array(dm2_cc)

    # DFT calculation
    mdft = pyscf.scf.UKS(mol)
    mdft.verbose = 4
    mdft.max_cycle = 200
    mdft.xc = "b3lyp"
    # get_veff_modified_uks(mdft, modeldict, lambda_rho=1, dm_tar=dm1_cc)
    mdft.kernel(mf.make_rdm1())
    if args.check_convergence and not mdft.converged:
        raise ValueError("UKS not converged.")
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    e_dft = mdft.energy_tot(dm1_dft)
    gdft = mdft.Gradients()
    grad_dft = gdft.kernel()

    # Compare CCSD and DFT
    print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} (CCSD vs DFT)")
    cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
    dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
    print(f"{np.linalg.norm(cc_dipole - dft_dipole)} (CCSD vs DFT)")
    energy_train = e_cc - e_dft
    grad_cc_train = grad_cc - grad_dft

    # Calculate the (exchange-correlation energy - DFT energy) on the grids and the grad to force matrix
    error_energy_dft, exc_cc_grids_dft, rho_cc, rho_dft, grad2force = get_dft_energy(
        mol,
        grids,
        dm1_dft,
        e_dft,
        mdft,
        mf,
        dm1_cc,
        dm1_cc_mo,
        dm2_cc,
        e_cc,
        evaluate=evaluate,
    )

    grad_mat = np.array(
        [
            0.08 * np.ones(len(grids.coords)),
            0.19 * np.ones(len(grids.coords)),
            0.72 * np.ones(len(grids.coords)),
            0.81 * np.ones(len(grids.coords)),
        ]
    )
    force = np.einsum(
        "mp,impx->ix",
        grad_mat,
        grad2force,
        optimize=True,
    )
    get_veff_grad_modified_zeros(gdft)
    grad_dft_zeros = gdft.kernel()
    print("Error force DFT: ", np.linalg.norm(force - (grad_dft - grad_dft_zeros)))

    # Generate input data
    rho_cube_dft = grids.gen_cube_rho_uks(rho_dft, mdft._numint, dm1_dft)
    rho_cube_cc = grids.gen_cube_rho_uks(rho_cc, mdft._numint, dm1_cc)
    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        mol=mol.tostring(format="xyz"),
        charge=mol.charge,
        spin=mol.spin,
        e_cc=e_cc,
        energy_train=energy_train,
        dm1_cc=dm1_cc,
        rho_cube_cc=rho_cube_cc,
        rho_cube_dft=rho_cube_dft,
        weights=grids.weights,
        exc_cc_grids=exc_cc_grids_dft,
        error_energy=error_energy_dft,
        grad2force=grad2force,
        grad_cc_train=grad_cc_train,
    )
