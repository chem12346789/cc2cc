# pylint: disable=W0212
import gc
from math import erf

import numpy as np
import opt_einsum as oe
import torch

import pyscf
from pyscf import lib
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
from pyscf.cc import uccsd_t
from pyscf.cc import uccsd_rdm
from pyscf.cc.uccsd_t_rdm import _gamma1_intermediates as u_gamma1_intermediates
from pyscf.cc.uccsd_t_rdm import _gamma2_intermediates as u_gamma2_intermediates
from pyscf.grad import uccsd_t as uccsd_t_grad
from pyscf.grad import uccsd as uccsd_grad

from cc2cc.gen_cc import block_loop_rdm2

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils.modelscf_uks import get_veff_grad_modified_zeros


def get_dft_energy(
    mol,
    grids,
    mdft,
    mf,
    dm1_hf,
    dm1_dft,
    dm1_cc,
    dm1_cc_mo,
    dm2_cc,
    evaluate=False,
):
    """
    Calculate the (exchange-correlation energy - DFT energy) on the grids.
    """
    backends = "torch" if torch.cuda.is_available() else "numpy"
    print(f"Using backend: {backends} for get_dft_energy\n")
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
    ao_1 = ao_value[1:4]

    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
    ]
    rho_hf = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_hf[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_hf[1], xctype="GGA"),
    ]

    if evaluate:
        return rho_dft, {}
    else:
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

        rho_cc = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA"),
        ]
        exc_dft_grids = pyscf.dft.libxc.eval_xc(
            "b3lyp", rho_dft, spin=1 if mol.spin else 0
        )[0] * (rho_dft[0][0] + rho_dft[1][0])

        # exchange part from dm2_cc
        print("Start exchange part...")
        exc_cc_grids = np.zeros_like(exc_dft_grids)
        for (
            i_slice,
            j_slice,
            k_slice,
            l_slice,
            nao_slice_i,
            nao_slice_j,
            nao_slice_k,
            nao_slice_l,
        ) in block_loop_rdm2(mol.nao):
            dm12_cc = (
                0.5 * dm2_cc[0][i_slice, j_slice, k_slice, l_slice]
                + 0.5 * dm2_cc[1][i_slice, j_slice, k_slice, l_slice]
                + 0.5
                * (dm2_cc[1].transpose(2, 3, 0, 1))[i_slice, j_slice, k_slice, l_slice]
                + 0.5 * dm2_cc[2][i_slice, j_slice, k_slice, l_slice]
                - 0.5
                * oe.contract(
                    "pq,rs->pqrs",
                    dm1_cc[0][i_slice, j_slice] + dm1_cc[1][i_slice, j_slice],
                    dm1_cc[0][k_slice, l_slice] + dm1_cc[1][k_slice, l_slice],
                )
            )

            expr_rinv_dm2_r = oe.contract_expression(
                "ijkl,i,j,kl->",
                dm12_cc,
                (nao_slice_i,),
                (nao_slice_j,),
                (nao_slice_k, nao_slice_l),
                constants=[0],
                optimize="optimal",
            )

            for i, coord in enumerate(grids.coords):
                if i * 10 % len(grids.coords) == 0:
                    print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
                ao_0_i = ao_value[0][i]
                with mol.with_rinv_origin(coord):
                    rinv = mol.intor("int1e_rinv")
                    exc_cc_grids[i] += expr_rinv_dm2_r(
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                        backend=backends,
                    )
            del expr_rinv_dm2_r, dm12_cc
            gc.collect()
            torch.cuda.empty_cache()
        print("Exchange part done.\n")

        # K part from dm1_dft
        # exchange part
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[0], dm1_dft[0])
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[1], dm1_dft[1])
        print("Start K part...")
        exc_k_dft_grids = np.zeros_like(exc_dft_grids)
        for (
            i_slice,
            j_slice,
            k_slice,
            l_slice,
            nao_slice_i,
            nao_slice_j,
            nao_slice_k,
            nao_slice_l,
        ) in block_loop_rdm2(mol.nao):
            dm12_cc = oe.contract(
                "pr,qs->pqrs",
                dm1_dft[0][i_slice, k_slice],
                dm1_dft[0][j_slice, l_slice],
            ) + oe.contract(
                "pr,qs->pqrs",
                dm1_dft[1][i_slice, k_slice],
                dm1_dft[1][j_slice, l_slice],
            )
            expr_rinv_dm2_r = oe.contract_expression(
                "ijkl,i,j,kl->",
                dm12_cc,
                (nao_slice_i,),
                (nao_slice_j,),
                (nao_slice_k, nao_slice_l),
                constants=[0],
                optimize="optimal",
            )

            for i, coord in enumerate(grids.coords):
                if i * 10 % len(grids.coords) == 0:
                    print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
                ao_0_i = ao_value[0][i]
                with mol.with_rinv_origin(coord):
                    rinv = mol.intor("int1e_rinv")
                    exc_k_dft_grids[i] += -0.1 * expr_rinv_dm2_r(
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                        backend=backends,
                    )
            del expr_rinv_dm2_r, dm12_cc
            gc.collect()
            torch.cuda.empty_cache()
        print("Exchange part done.\n")

        # K part from dm1_hf
        # exchange part
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[0], dm1_hf[0])
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[1], dm1_hf[1])
        print("Start K part from HF...")
        exc_k_hf_grids = np.zeros_like(exc_dft_grids)
        for (
            i_slice,
            j_slice,
            k_slice,
            l_slice,
            nao_slice_i,
            nao_slice_j,
            nao_slice_k,
            nao_slice_l,
        ) in block_loop_rdm2(mol.nao):
            dm12_cc = oe.contract(
                "pr,qs->pqrs",
                dm1_hf[0][i_slice, k_slice],
                dm1_hf[0][j_slice, l_slice],
            ) + oe.contract(
                "pr,qs->pqrs",
                dm1_hf[1][i_slice, k_slice],
                dm1_hf[1][j_slice, l_slice],
            )
            expr_rinv_dm2_r = oe.contract_expression(
                "ijkl,i,j,kl->",
                dm12_cc,
                (nao_slice_i,),
                (nao_slice_j,),
                (nao_slice_k, nao_slice_l),
                constants=[0],
                optimize="optimal",
            )

            for i, coord in enumerate(grids.coords):
                if i * 10 % len(grids.coords) == 0:
                    print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
                ao_0_i = ao_value[0][i]
                with mol.with_rinv_origin(coord):
                    rinv = mol.intor("int1e_rinv")
                    exc_k_hf_grids[i] += -0.5 * expr_rinv_dm2_r(
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                        backend=backends,
                    )
            del expr_rinv_dm2_r
            gc.collect()
            torch.cuda.empty_cache()

        # hatree part from dm1_cc
        print("Start hatree part...")
        hatree_cc_grids = np.zeros_like(exc_dft_grids)
        int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_cc[0] + dm1_cc[1],
        )
        hatree_cc_grids += 0.5 * vele * (rho_cc[0][0] + rho_cc[1][0])

        # hatree part from dm1_dft
        hatree_dft_grids = np.zeros_like(exc_dft_grids)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_dft[0] + dm1_dft[1],
        )
        hatree_dft_grids += 0.5 * vele * (rho_dft[0][0] + rho_dft[1][0])
        print("Hatree part done.\n")

        hatree_hf_grids = np.zeros_like(exc_dft_grids)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_hf[0] + dm1_hf[1],
        )
        hatree_hf_grids += 0.5 * vele * (rho_hf[0][0] + rho_hf[1][0])

        # kinetic part
        kin_cc_grids = np.zeros_like(exc_dft_grids)
        kin_dft_grids = np.zeros_like(exc_dft_grids)
        kin_hf_grids = np.zeros_like(exc_dft_grids)
        kinl_cc_grids = np.zeros_like(exc_dft_grids)
        kinl_dft_grids = np.zeros_like(exc_dft_grids)
        kinl_hf_grids = np.zeros_like(exc_dft_grids)
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
                kin_cc_grids -= part * eigs_e_dm1[i] / 2

                partl = oe.contract(
                    "xpm,m,n,xpn->p",
                    ao_1,
                    eigs_v_dm1[:, i],
                    eigs_v_dm1[:, i],
                    ao_1,
                )
                kinl_cc_grids += partl * eigs_e_dm1[i] / 2

            for i in range(mol.nelec[i_spin]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_value[0],
                    dft_mo_coeff[i_spin][:, i],
                    dft_mo_coeff[i_spin][:, i],
                    ao_2_diag,
                )
                kin_dft_grids -= part / 2

                partl = oe.contract(
                    "xpm,m,n,xpn->p",
                    ao_1,
                    dft_mo_coeff[i_spin][:, i],
                    dft_mo_coeff[i_spin][:, i],
                    ao_1,
                )
                kinl_dft_grids += partl / 2

            for i in range(mol.nelec[i_spin]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_value[0],
                    mf_mo_coeff[i_spin][:, i],
                    mf_mo_coeff[i_spin][:, i],
                    ao_2_diag,
                )
                kin_hf_grids -= part / 2

                partl = oe.contract(
                    "xpm,m,n,xpn->p",
                    ao_1,
                    mf_mo_coeff[i_spin][:, i],
                    mf_mo_coeff[i_spin][:, i],
                    ao_1,
                )
                kinl_hf_grids += partl / 2

        # nuclear part
        nuc_cc_grids = np.zeros_like(exc_dft_grids)
        nuc_dft_grids = np.zeros_like(exc_dft_grids)
        nuc_hf_grids = np.zeros_like(exc_dft_grids)
        nuc_erf_cc_grids = np.zeros_like(exc_dft_grids)
        nuc_erf_dft_grids = np.zeros_like(exc_dft_grids)
        nuc_erf_hf_grids = np.zeros_like(exc_dft_grids)
        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-8:
                    nuc_cc_grids[i] -= (
                        (rho_cc[0][0][i] + rho_cc[1][0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_dft_grids[i] -= (
                        (rho_dft[0][0][i] + rho_dft[1][0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_hf_grids[i] -= (
                        (rho_hf[0][0][i] + rho_hf[1][0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_erf_cc_grids[i] -= (
                        (rho_cc[0][0][i] + rho_cc[1][0][i])
                        * mol.atom_charges()[i_atom]
                        * erf(1e2 * distance)
                        / distance
                    )
                    nuc_erf_dft_grids[i] -= (
                        (rho_dft[0][0][i] + rho_dft[1][0][i])
                        * mol.atom_charges()[i_atom]
                        * erf(1e2 * distance)
                        / distance
                    )
                    nuc_erf_hf_grids[i] -= (
                        (rho_hf[0][0][i] + rho_hf[1][0][i])
                        * mol.atom_charges()[i_atom]
                        * erf(1e2 * distance)
                        / distance
                    )
        real_nuc_cc = np.einsum(
            "pq,pq->", mol.intor("int1e_nuc"), dm1_cc[0] + dm1_cc[1]
        )
        real_nuc_dft = np.einsum(
            "pq,pq->", mol.intor("int1e_nuc"), dm1_dft[0] + dm1_dft[1]
        )
        real_nuc_hf = np.einsum(
            "pq,pq->", mol.intor("int1e_nuc"), dm1_hf[0] + dm1_hf[1]
        )
        nuc_erf_cc_scale = real_nuc_cc / np.sum(nuc_erf_cc_grids * grids.weights)
        nuc_erf_cc_grids *= nuc_erf_cc_scale
        nuc_erf_dft_scale = real_nuc_dft / np.sum(nuc_erf_dft_grids * grids.weights)
        nuc_erf_dft_grids *= nuc_erf_dft_scale
        nuc_erf_hf_scale = real_nuc_hf / np.sum(nuc_erf_hf_grids * grids.weights)
        nuc_erf_hf_grids *= nuc_erf_hf_scale
        print(
            "Nuclear scale factors (cc/dft/hf):",
            nuc_erf_cc_scale,
            nuc_erf_dft_scale,
            nuc_erf_hf_scale,
        )
        print("Nuclear part done.\n")

        tol_cc_grids = exc_cc_grids + hatree_cc_grids + kin_cc_grids + nuc_cc_grids
        tol_dft_grids = (
            exc_dft_grids
            + exc_k_dft_grids
            + hatree_dft_grids
            + kin_dft_grids
            + nuc_dft_grids
        )
        tol_delta_grids = tol_cc_grids - tol_dft_grids

        return (
            rho_dft,
            {
                "exc_cc_grids": exc_cc_grids,
                "hatree_cc_grids": hatree_cc_grids,
                "kin_cc_grids": kin_cc_grids,
                "kinl_cc_grids": kinl_cc_grids,
                "nuc_cc_grids": nuc_cc_grids,
                "nuc_erf_cc_grids": nuc_erf_cc_grids,
                "exc_dft_grids": exc_dft_grids,
                "exc_k_dft_grids": exc_k_dft_grids,
                "hatree_dft_grids": hatree_dft_grids,
                "kin_dft_grids": kin_dft_grids,
                "kinl_dft_grids": kinl_dft_grids,
                "nuc_dft_grids": nuc_dft_grids,
                "nuc_erf_dft_grids": nuc_erf_dft_grids,
                "exc_k_hf_grids": exc_k_hf_grids,
                "hatree_hf_grids": hatree_hf_grids,
                "kin_hf_grids": kin_hf_grids,
                "kinl_hf_grids": kinl_hf_grids,
                "nuc_hf_grids": nuc_hf_grids,
                "nuc_erf_hf_grids": nuc_erf_hf_grids,
                "tol_delta_grids": tol_delta_grids,
                "grad2force": grad2force,
            },
        )


def ucc(mol, grids, name, args, evaluate=False):
    """
    Generate data for the UCCSD method.
    """
    print(f"Generate data for {name}, spin {mol.spin}")

    # UHF calculation
    mf = pyscf.scf.UHF(mol)
    mf.max_cycle = 2500
    mf.verbose = 4
    mf.kernel()
    if args.check_convergence and not mf.converged:
        pyscf.scf.addons.dynamic_level_shift_(mf, factor=0.5)
        mf.kernel()
    dm1_hf = mf.make_rdm1(ao_repr=True)
    e_hf = mf.e_tot

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
    e_dft = mdft.e_tot

    mdft_d3bj = pyscf.scf.UKS(mol)
    mdft_d3bj.verbose = 4
    mdft_d3bj.max_cycle = 200
    mdft_d3bj.xc = "b3lyp-d3bj"
    mdft_d3bj.kernel(dm1_dft)
    e_dft_d3bj = mdft_d3bj.e_tot
    print(f"DFT-D3BJ correct energy: {e_dft_d3bj - e_dft}")

    # UCCSD calculation
    mycc = pyscf.cc.UCCSD(mf)
    mycc.verbose = 9
    mycc.direct = False
    _, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = uccsd_t.kernel(mycc, eris, t1, t2)
    e_cc = mycc.e_tot + e3ref
    print(f"UCCSD(T) energy: {e_cc}")
    energy_train = e_cc - e_dft

    if evaluate:
        dm1_cc = None
        dm1_cc_mo = None
        dm2_cc = None
        grad_cc = None
        del t1, t2, eris, mycc
        gc.collect()
    else:
        l1, l2 = uccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
        d1 = u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        d2 = u_gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
        # CC gradient
        if mol.natm == 1:
            grad_cc = np.zeros((mol.natm, 3))  # Fallback to zero gradients
        elif mol.nelectron == 1:
            ghf = pyscf.grad.uhf.Gradients(mf)
            grad_cc = ghf.kernel()
        else:
            gcc = uccsd_t_grad.Gradients(mycc)
            grad_cc = gcc.kernel(t1, t2, l1, l2, eris=eris)
        del t1, t2, l1, l2
        gc.collect()

        dm1_cc_mo = uccsd_rdm._make_rdm1(mycc, d1, with_frozen=True, ao_repr=False)
        mo_a, mo_b = mycc.mo_coeff
        dm1_cc = np.array(
            [
                np.einsum("pi,ij,qj->pq", mo_a, dm1_cc_mo[0], mo_a),
                np.einsum("pi,ij,qj->pq", mo_b, dm1_cc_mo[1], mo_b),
            ]
        )
        dm2_cc = uccsd_rdm._make_rdm2(
            mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=True
        )
        dm1_cc_mo = np.array(dm1_cc_mo)
        dm2_cc = np.array(dm2_cc)
        del d1, d2, eris, mycc
        gc.collect()

        # Compare CCSD and DFT
        print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} (CCSD vs DFT)")
        cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
        dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
        print(f"{np.linalg.norm(cc_dipole - dft_dipole)} (CCSD vs DFT)")

    # Calculate the (exchange-correlation energy - DFT energy) on the grids and the grad to force matrix
    rho_dft, data_dict = get_dft_energy(
        mol,
        grids,
        mdft,
        mf,
        dm1_hf,
        dm1_dft,
        dm1_cc,
        dm1_cc_mo,
        dm2_cc,
        evaluate=evaluate,
    )

    if "tol_delta_grids" in data_dict:
        error = np.sum(data_dict["tol_delta_grids"] * grids.weights) - energy_train
        print(f"Error: {AU2KCALMOL * error}")
        error_hf = np.sum(
            (
                (
                    data_dict["exc_cc_grids"]
                    + data_dict["hatree_cc_grids"]
                    + data_dict["kin_cc_grids"]
                    + data_dict["nuc_cc_grids"]
                )
                - (
                    data_dict["exc_k_hf_grids"]
                    + data_dict["hatree_hf_grids"]
                    + data_dict["kin_hf_grids"]
                    + data_dict["nuc_hf_grids"]
                )
            )
            * grids.weights
        ) - (e_cc - e_hf)
        print(f"Error HF part: {AU2KCALMOL * error_hf}")

        error_exc_lerf = (
            np.sum(
                (
                    (
                        data_dict["exc_cc_grids"]
                        + data_dict["hatree_cc_grids"]
                        + data_dict["kinl_cc_grids"]
                        + data_dict["nuc_erf_cc_grids"]
                    )
                    - (
                        data_dict["exc_dft_grids"]
                        + data_dict["exc_k_dft_grids"]
                        + data_dict["hatree_dft_grids"]
                        + data_dict["kinl_dft_grids"]
                        + data_dict["nuc_erf_dft_grids"]
                    )
                )
                * grids.weights
            )
            - energy_train
        )
        print(f"Error exc_lerf part: {AU2KCALMOL * error_exc_lerf}")

    if "grad2force" in data_dict and grad_cc is not None:
        # HF gradient
        ghf = pyscf.grad.uhf.Gradients(mf)
        grad_hf = ghf.kernel()

        # DFT gradient
        gdft = mdft.Gradients()
        grad_dft = gdft.kernel()

        data_dict["grad_cc_train"] = grad_cc - grad_dft
        data_dict["grad_hf"] = grad_hf
        data_dict["grad_dft"] = grad_dft
        data_dict["grad_cc"] = grad_cc

        # Test force
        grad_mat = np.array(
            [
                0.08 * np.ones(len(grids.coords)),
                0.19 * np.ones(len(grids.coords)),
                0.72 * np.ones(len(grids.coords)),
                0.81 * np.ones(len(grids.coords)),
            ]
        )
        force = np.einsum(
            "mp,impx->ix", grad_mat, data_dict["grad2force"], optimize=True
        )
        get_veff_grad_modified_zeros(gdft)
        grad_dft_zeros = gdft.kernel()
        print("Error force DFT: ", np.linalg.norm(force - (grad_dft - grad_dft_zeros)))

    # Generate input data
    rho_cube_dft = grids.gen_cube_rho_uks(rho_dft, mdft._numint, dm1_dft)

    data_dict.update(
        {
            "mol": mol.tostring(format="xyz"),
            "charge": mol.charge,
            "spin": mol.spin,
            "e_cc": e_cc,
            "e_dft": e_dft,
            "e_hf": e_hf,
            "dm1_hf": dm1_hf,
            "dm1_dft": dm1_dft,
            "dm1_cc": dm1_cc,
            "e_dft_d3bj": e_dft_d3bj,
            "energy_train": energy_train,
            "rho_cube_dft": rho_cube_dft,
            "weights": grids.weights,
        }
    )
    np.savez_compressed(DATA_PATH / f"data_{name}.npz", **data_dict)
