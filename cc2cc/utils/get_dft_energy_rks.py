import gc
from math import erf
from itertools import product

import numpy as np
import opt_einsum as oe
import torch

import pyscf
import pyscf.dft


def block_loop_rdm2(nao):
    """
    Generate slices for block processing of the 4-index 2-RDM.
    50 slices per dimension.
    """
    n_slices = 125
    n_batchs = nao // n_slices + 1
    total_batches = n_batchs**4
    print(
        f"Total batches for rdm2: {total_batches}. n_batchs: {n_batchs}, n_slices: {n_slices}, will consume about {n_slices**4 * 8 / 1024**3:.2f} GB memory."
    )
    for i_batch, j_batch, k_batch, l_batch in product(
        range(n_batchs),
        range(n_batchs),
        range(n_batchs),
        range(n_batchs),
    ):
        print(
            f"Processing batch {i_batch * n_batchs**3 + j_batch * n_batchs**2 + k_batch * n_batchs + l_batch + 1} / {total_batches}",
            flush=True,
        )
        nao_slice_i = n_slices if i_batch != n_batchs - 1 else nao - n_slices * i_batch
        nao_slice_j = n_slices if j_batch != n_batchs - 1 else nao - n_slices * j_batch
        nao_slice_k = n_slices if k_batch != n_batchs - 1 else nao - n_slices * k_batch
        nao_slice_l = n_slices if l_batch != n_batchs - 1 else nao - n_slices * l_batch

        i_slice = slice(n_slices * i_batch, n_slices * i_batch + nao_slice_i)
        j_slice = slice(n_slices * j_batch, n_slices * j_batch + nao_slice_j)
        k_slice = slice(n_slices * k_batch, n_slices * k_batch + nao_slice_k)
        l_slice = slice(n_slices * l_batch, n_slices * l_batch + nao_slice_l)
        yield i_slice, j_slice, k_slice, l_slice, nao_slice_i, nao_slice_j, nao_slice_k, nao_slice_l


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

        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")

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
            dm12_cc = 0.5 * dm2_cc[
                i_slice, j_slice, k_slice, l_slice
            ] - 0.5 * oe.contract(
                "pq,rs->pqrs", dm1_cc[i_slice, j_slice], dm1_cc[k_slice, l_slice]
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
            dm1_cc,
        )
        hatree_cc_grids += 0.5 * vele * rho_cc[0]

        # kinetic part
        kin_cc_grids = np.zeros_like(grids.weights)
        kinl_cc_grids = np.zeros_like(grids.weights)
        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
        eigs_v_dm1 = mf_mo_coeff @ eigs_v_dm1
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

        # nuclear part
        nuc_cc_grids = np.zeros_like(grids.weights)
        nuc_erf_cc_grids = np.zeros_like(grids.weights)
        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-8:
                    nuc_cc_grids[i] -= (
                        rho_cc[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_erf_cc_grids[i] -= (
                        rho_cc[0][i]
                        * erf(1e2 * distance)
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
        real_nuc_cc = np.einsum("pq,pq->", mol.intor("int1e_nuc"), dm1_cc)
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

    rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")

    if evaluate:
        return {}
    else:
        dft_mo_coeff = mdft.mo_coeff

        exc_dft_grids = pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0]

        # K part from dm1_dft
        # exchange part
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[0], dm1_dft[0])
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[1], dm1_dft[1])
        # alpha is 0.2 in b3lyp
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
            expr_rinv_dm2_r = oe.contract_expression(
                "ik,jl,i,j,kl->",
                dm1_dft[i_slice, k_slice],
                dm1_dft[j_slice, l_slice],
                (nao_slice_i,),
                (nao_slice_j,),
                (nao_slice_k, nao_slice_l),
                constants=[0, 1],
                optimize="optimal",
            )

            for i, coord in enumerate(grids.coords):
                if i * 10 % len(grids.coords) == 0:
                    print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
                ao_0_i = ao_value[0][i]
                with mol.with_rinv_origin(coord):
                    rinv = mol.intor("int1e_rinv")
                    exc_k_dft_grids[i] += -0.05 * expr_rinv_dm2_r(
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                        backend=backends,
                    )
            del expr_rinv_dm2_r
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
            dm1_dft,
        )
        hatree_dft_grids += 0.5 * vele * rho_dft[0]
        print("Hatree part done.\n")

        # kinetic part
        print("Start kinetic part...")
        kin_dft_grids = np.zeros_like(grids.weights)
        kinl_dft_grids = np.zeros_like(grids.weights)
        for i in range(mol.nelec[0]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                dft_mo_coeff[:, i],
                dft_mo_coeff[:, i],
                ao_2_diag,
            )
            kin_dft_grids -= part

            partl = oe.contract(
                "xpm,m,n,xpn->p",
                ao_1,
                dft_mo_coeff[:, i],
                dft_mo_coeff[:, i],
                ao_1,
            )
            kinl_dft_grids += partl
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
                        rho_dft[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_erf_dft_grids[i] -= (
                        rho_dft[0][i]
                        * erf(1e2 * distance)
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
        real_nuc_dft = np.einsum("pq,pq->", mol.intor("int1e_nuc"), dm1_dft)
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
    Calculate the (exchange-correlation energy - DFT energy) on the grids.
    """
    backends = "torch" if torch.cuda.is_available() else "numpy"
    print(f"Using backend: {backends} for get_dft_energy\n")
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]
    ao_1 = ao_value[1:4]

    rho_hf = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_hf, xctype="GGA")

    if evaluate:
        return {}
    else:
        mf_mo_coeff = mf.mo_coeff

        # K part from dm1_hf
        # exchange part
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[0], dm1_hf[0])
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[1], dm1_hf[1])
        # coefficient = - 2 * 0.5 * 0.5 * 0.5 = -0.25
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
            expr_rinv_dm2_r = oe.contract_expression(
                "ik,jl,i,j,kl->",
                dm1_hf[i_slice, k_slice],
                dm1_hf[j_slice, l_slice],
                (nao_slice_i,),
                (nao_slice_j,),
                (nao_slice_k, nao_slice_l),
                constants=[0, 1],
                optimize="optimal",
            )

            for i, coord in enumerate(grids.coords):
                if i * 10 % len(grids.coords) == 0:
                    print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
                ao_0_i = ao_value[0][i]
                with mol.with_rinv_origin(coord):
                    rinv = mol.intor("int1e_rinv")
                    exc_k_hf_grids[i] += -0.25 * expr_rinv_dm2_r(
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
        int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
        hatree_hf_grids = np.zeros_like(grids.weights)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_hf,
        )
        hatree_hf_grids += 0.5 * vele * rho_hf[0]
        print("Hatree part done.\n")

        # kinetic part
        print("Start kinetic part...")
        kin_hf_grids = np.zeros_like(grids.weights)
        kinl_hf_grids = np.zeros_like(grids.weights)
        for i in range(mol.nelec[0]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                mf_mo_coeff[:, i],
                mf_mo_coeff[:, i],
                ao_2_diag,
            )
            kin_hf_grids -= part

            partl = oe.contract(
                "xpm,m,n,xpn->p",
                ao_1,
                mf_mo_coeff[:, i],
                mf_mo_coeff[:, i],
                ao_1,
            )
            kinl_hf_grids += partl
        print("Kinetic part done.\n")

        # nuclear part
        print("Start nuclear part...")
        nuc_hf_grids = np.zeros_like(grids.weights)
        nuc_erf_hf_grids = np.zeros_like(grids.weights)
        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-8:
                    nuc_hf_grids[i] -= (
                        rho_hf[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_erf_hf_grids[i] -= (
                        rho_hf[0][i]
                        * erf(1e2 * distance)
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
        real_nuc_hf = np.einsum("pq,pq->", mol.intor("int1e_nuc"), dm1_hf)
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
