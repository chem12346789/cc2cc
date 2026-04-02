import gc
from math import erf
from itertools import product

import numpy as np
import opt_einsum as oe
import torch

import pyscf
import pyscf.dft

from cc2cc.utils.get_dft_energy_rks import block_loop_rdm2


def get_cc_energy(
    mol,
    grids,
    mf,
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
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]
    ao_1 = ao_value[1:4]

    if evaluate:
        return {}
    else:
        mf_mo_coeff = mf.mo_coeff

        rho_cc = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA"),
        ]

        # exchange part from dm2_cc
        print("Start exchange part...")
        exc_cc_grids = np.zeros_like(grids.weights)
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

        # hatree part from dm1_cc
        print("Start hatree part...")
        hatree_cc_grids = np.zeros_like(grids.weights)
        int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_cc[0] + dm1_cc[1],
        )
        hatree_cc_grids += 0.5 * vele * (rho_cc[0][0] + rho_cc[1][0])
        print("Hatree part done.\n")

        # kinetic part
        print("Start kinetic part...")
        kin_cc_grids = np.zeros_like(grids.weights)
        kinl_cc_grids = np.zeros_like(grids.weights)
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
        print("Kinetic part done.\n")

        # nuclear part
        print("Start nuclear part...")
        nuc_cc_grids = np.zeros_like(grids.weights)
        nuc_erf_cc_grids = np.zeros_like(grids.weights)
        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-8:
                    nuc_cc_grids[i] -= (
                        (rho_cc[0][0][i] + rho_cc[1][0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_erf_cc_grids[i] -= (
                        (rho_cc[0][0][i] + rho_cc[1][0][i])
                        * mol.atom_charges()[i_atom]
                        * erf(1e2 * distance)
                        / distance
                    )
        real_nuc_cc = np.einsum(
            "pq,pq->", mol.intor("int1e_nuc"), dm1_cc[0] + dm1_cc[1]
        )
        nuc_erf_cc_scale = real_nuc_cc / np.sum(nuc_erf_cc_grids * grids.weights)
        nuc_erf_cc_grids *= nuc_erf_cc_scale
        print("Nuclear scale factors (cc/dft/hf):", nuc_erf_cc_scale)
        print("Nuclear part done.\n")

        tol_cc_grids = exc_cc_grids + hatree_cc_grids + kin_cc_grids + nuc_cc_grids

        return {
            "exc_cc_grids": exc_cc_grids,
            "hatree_cc_grids": hatree_cc_grids,
            "kin_cc_grids": kin_cc_grids,
            "kinl_cc_grids": kinl_cc_grids,
            "nuc_cc_grids": nuc_cc_grids,
            "nuc_erf_cc_grids": nuc_erf_cc_grids,
            "tol_cc_grids": tol_cc_grids,
        }


def get_dft_energy(
    mol,
    grids,
    mdft,
    dm1_dft,
    evaluate=False,
):
    """
    Calculate the (exchange-correlation energy - DFT energy) on the grids.
    """
    backends = "torch" if torch.cuda.is_available() else "numpy"
    print(f"Using backend: {backends} for get_dft_energy\n")
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]
    ao_1 = ao_value[1:4]

    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
    ]

    if evaluate:
        return {}
    else:
        dft_mo_coeff = mdft.mo_coeff

        exc_dft_grids = pyscf.dft.libxc.eval_xc(
            "b3lyp", rho_dft, spin=1 if mol.spin else 0
        )[0] * (rho_dft[0][0] + rho_dft[1][0])

        # K part from dm1_dft
        # exchange part
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[0], dm1_dft[0])
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[1], dm1_dft[1])
        print("Start K part...")
        exc_k_dft_grids = np.zeros_like(grids.weights)
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

        # hatree part from dm1_dft
        print("Start hatree part...")
        int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
        hatree_dft_grids = np.zeros_like(grids.weights)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_dft[0] + dm1_dft[1],
        )
        hatree_dft_grids += 0.5 * vele * (rho_dft[0][0] + rho_dft[1][0])
        print("Hatree part done.\n")

        # kinetic part
        print("Start kinetic part...")
        kin_dft_grids = np.zeros_like(grids.weights)
        kinl_dft_grids = np.zeros_like(grids.weights)
        for i_spin in range(2):
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
        print("Kinetic part done.\n")

        # nuclear part
        print("Start nuclear part...")
        nuc_dft_grids = np.zeros_like(grids.weights)
        nuc_erf_dft_grids = np.zeros_like(grids.weights)
        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-8:
                    nuc_dft_grids[i] -= (
                        (rho_dft[0][0][i] + rho_dft[1][0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_erf_dft_grids[i] -= (
                        (rho_dft[0][0][i] + rho_dft[1][0][i])
                        * mol.atom_charges()[i_atom]
                        * erf(1e2 * distance)
                        / distance
                    )
        real_nuc_dft = np.einsum(
            "pq,pq->", mol.intor("int1e_nuc"), dm1_dft[0] + dm1_dft[1]
        )
        nuc_erf_dft_scale = real_nuc_dft / np.sum(nuc_erf_dft_grids * grids.weights)
        nuc_erf_dft_grids *= nuc_erf_dft_scale
        print("Nuclear scale factors (cc/dft/hf):", nuc_erf_dft_scale)
        print("Nuclear part done.\n")

        tol_dft_grids = (
            exc_dft_grids
            + exc_k_dft_grids
            + hatree_dft_grids
            + kin_dft_grids
            + nuc_dft_grids
        )

        return {
            "exc_dft_grids": exc_dft_grids,
            "exc_k_dft_grids": exc_k_dft_grids,
            "hatree_dft_grids": hatree_dft_grids,
            "kin_dft_grids": kin_dft_grids,
            "kinl_dft_grids": kinl_dft_grids,
            "nuc_dft_grids": nuc_dft_grids,
            "nuc_erf_dft_grids": nuc_erf_dft_grids,
            "tol_dft_grids": tol_dft_grids,
        }


def get_hf_energy(
    mol,
    grids,
    mf,
    dm1_hf,
    evaluate=False,
):
    """
    Calculate the (exchange-correlation energy - HF energy) on the grids.
    """
    backends = "torch" if torch.cuda.is_available() else "numpy"
    print(f"Using backend: {backends} for get_hf_energy\n")
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]
    ao_1 = ao_value[1:4]

    rho_hf = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_hf[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_hf[1], xctype="GGA"),
    ]

    if evaluate:
        return {}
    else:
        mf_mo_coeff = mf.mo_coeff

        # K part from dm1_hf
        # exchange part
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[0], dm1_hf[0])
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[1], dm1_hf[1])
        print("Start K part from HF...")
        exc_k_hf_grids = np.zeros_like(grids.weights)
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
        print("Exchange part done.\n")

        # hatree part from dm1_hf
        print("Start hatree part from HF...")
        int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
        hatree_hf_grids = np.zeros_like(grids.weights)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_hf[0] + dm1_hf[1],
        )
        hatree_hf_grids += 0.5 * vele * (rho_hf[0][0] + rho_hf[1][0])
        print("Hatree part done.\n")

        # kinetic part
        print("Start kinetic part from HF...")
        kin_hf_grids = np.zeros_like(grids.weights)
        kinl_hf_grids = np.zeros_like(grids.weights)
        for i_spin in range(2):
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
        print("Kinetic part done.\n")

        # nuclear part
        print("Start nuclear part from HF...")
        nuc_hf_grids = np.zeros_like(grids.weights)
        nuc_erf_hf_grids = np.zeros_like(grids.weights)
        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-8:
                    nuc_hf_grids[i] -= (
                        (rho_hf[0][0][i] + rho_hf[1][0][i])
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_erf_hf_grids[i] -= (
                        (rho_hf[0][0][i] + rho_hf[1][0][i])
                        * mol.atom_charges()[i_atom]
                        * erf(1e2 * distance)
                        / distance
                    )
        real_nuc_hf = np.einsum(
            "pq,pq->", mol.intor("int1e_nuc"), dm1_hf[0] + dm1_hf[1]
        )
        nuc_erf_hf_scale = real_nuc_hf / np.sum(nuc_erf_hf_grids * grids.weights)
        nuc_erf_hf_grids *= nuc_erf_hf_scale
        print("Nuclear scale factors (cc/dft/hf):", nuc_erf_hf_scale)
        print("Nuclear part done.\n")

        return {
            "exc_k_hf_grids": exc_k_hf_grids,
            "hatree_hf_grids": hatree_hf_grids,
            "kin_hf_grids": kin_hf_grids,
            "kinl_hf_grids": kinl_hf_grids,
            "nuc_hf_grids": nuc_hf_grids,
            "nuc_erf_hf_grids": nuc_erf_hf_grids,
        }
