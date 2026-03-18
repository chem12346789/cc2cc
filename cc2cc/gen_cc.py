# pylint: disable=W0212
import gc
from itertools import product
from math import erf

import numpy as np
import scipy
import opt_einsum as oe
import torch

import pyscf
from pyscf import lib

import pyscf.cc
from pyscf.cc import ccsd_t_lambda
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_rdm

# from pyscf.cc.ccsd_t_rdm import _gamma1_intermediates
from pyscf.cc.ccsd_t_rdm import _gamma2_intermediates
from pyscf.grad import ccsd as ccsd_grad

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils.modelscf_rks import get_veff_grad_modified_zeros
from cc2cc.utils.pyscf_ccsd_t_rdm import _gamma1_intermediates
from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_SIZE


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
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]
    ao_1 = ao_value[1:4]

    rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
    rho_hf = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_hf, xctype="GGA")

    if evaluate:
        return rho_dft, {}
    else:
        dft_mo_coeff = mdft.mo_coeff
        mf_mo_coeff = mf.mo_coeff

        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        exc_dft_grids = pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0] * rho_dft[0]

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

        # K part from dm1_dft
        # exchange part
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[0], dm1_dft[0])
        # - 0.5 * alpha * oe.contract("pr,qs->pqrs", dm1_dft[1], dm1_dft[1])
        # alpha is 0.2 in b3lyp
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

        # K part from dm1_hf
        # exchange part
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[0], dm1_hf[0])
        # - 0.5 * oe.contract("pr,qs->pqrs", dm1_hf[1], dm1_hf[1])
        # coefficient = - 2 * 0.5 * 0.5 * 0.5 = -0.25
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
        hatree_cc_grids = np.zeros_like(exc_dft_grids)
        int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_cc,
        )
        hatree_cc_grids += 0.5 * vele * rho_cc[0]

        # hatree part from dm1_dft
        hatree_dft_grids = np.zeros_like(exc_dft_grids)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_dft,
        )
        hatree_dft_grids += 0.5 * vele * rho_dft[0]
        print("Hatree part done.\n")

        # hatree part from dm1_hf
        hatree_hf_grids = np.zeros_like(exc_dft_grids)
        vele = np.einsum(
            "pij,ij->p",
            int1e_grids,
            dm1_hf,
        )
        hatree_hf_grids += 0.5 * vele * rho_hf[0]

        # kinetic part
        kin_cc_grids = np.zeros_like(exc_dft_grids)
        kin_dft_grids = np.zeros_like(exc_dft_grids)
        kin_hf_grids = np.zeros_like(exc_dft_grids)
        kinl_cc_grids = np.zeros_like(exc_dft_grids)
        kinl_dft_grids = np.zeros_like(exc_dft_grids)
        kinl_hf_grids = np.zeros_like(exc_dft_grids)
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
                        rho_cc[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_dft_grids[i] -= (
                        rho_dft[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_hf_grids[i] -= (
                        rho_hf[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_erf_cc_grids[i] -= (
                        rho_cc[0][i]
                        * erf(1e2 * distance)
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_erf_dft_grids[i] -= (
                        rho_dft[0][i]
                        * erf(1e2 * distance)
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
                    nuc_erf_hf_grids[i] -= (
                        rho_hf[0][i]
                        * erf(1e2 * distance)
                        * mol.atom_charges()[i_atom]
                        / distance
                    )
        real_nuc_cc = np.einsum("pq,pq->", mol.intor("int1e_nuc"), dm1_cc)
        real_nuc_dft = np.einsum("pq,pq->", mol.intor("int1e_nuc"), dm1_dft)
        real_nuc_hf = np.einsum("pq,pq->", mol.intor("int1e_nuc"), dm1_hf)
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

        return {
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
        }


def get_dft_grad(mol, grids, dm1_dft, data_dict, max_memory=8000):
    """
    Calculate the gradient of (exchange-correlation energy - DFT energy) on the grids.
    Note the max_memory is hard to predict (a large memory usage is due to grad2force and grad_mat), so just set it to a relative small value to avoid OOM.
    """
    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.verbose = 4
    mdft.kernel(dm1_dft)
    gdft = mdft.Gradients()
    grad_dft = gdft.kernel()
    get_veff_grad_modified_zeros(gdft)
    grad_dft_zeros = gdft.kernel()

    atmlst = range(mol.natm)
    grad2force = np.zeros(
        (
            len(atmlst),
            grids.input_level,
            len(grids.coords),
            EDGE_SIZE,
            EDGE_SIZE,
            EDGE_SIZE,
            3,
        )
    )

    rho_cube_dft = np.zeros((len(grids.coords), grids.input_level, EDGE_SIZE**3))
    gridcube_coords = np.zeros((len(grids.coords), EDGE_SIZE**3, 3))
    ao0_coords = np.zeros((len(grids.coords), EDGE_SIZE**3, dm1_dft.shape[-1]))
    ao4_coords = np.zeros(
        (grids.input_level, len(grids.coords), EDGE_SIZE**3, dm1_dft.shape[-1])
    )

    ni = mdft._numint
    step = int(max_memory * 1024**2 / (dm1_dft.shape[-1] * EDGE_SIZE**3 * 32 * 8))
    # 32 is the number of elements in the ao_array and ao_mat, 8 is the size of float64 in bytes
    print(f"Step size: {step}")
    for p0, p1 in lib.prange(0, len(grids.coords), step):
        if grids.screen_index is None:
            mask = None
        else:
            mask = grids.screen_index[p0:p1]
        coords_ = grids.coords[p0:p1]
        gridcube = grids.gen_cube(mol, dm1_dft, coords_, mask)
        rho_cube_dft_part, wv, ao_value = gridcube.gen_cube_rho_rks(
            ni, dm1_dft, ao_deriv=2, require_vxc=True
        )
        rho_cube_dft[p0:p1] = rho_cube_dft_part
        gridcube_coords[p0:p1] = gridcube.coords.reshape(len(coords_), EDGE_SIZE**3, 3)
        ao0_coords[p0:p1] = ao_value[0].reshape(
            len(coords_), EDGE_SIZE**3, dm1_dft.shape[-1]
        )

        wv = wv.reshape(gridcube.input_level, 4, len(gridcube.coords))
        wv[:, 0, :] *= 0.5
        ao4_coords[:, p0:p1] = np.einsum("xpu,ixp->ipu", ao_value[:4], wv).reshape(
            gridcube.input_level, len(coords_), EDGE_SIZE**3, dm1_dft.shape[-1]
        )

        ao_array = np.array([ao_value[0], ao_value[1], ao_value[2], ao_value[3]])
        ao_mat = np.array(
            [
                [ao_value[1], ao_value[2], ao_value[3]],
                [ao_value[4], ao_value[5], ao_value[6]],
                [ao_value[5], ao_value[7], ao_value[8]],
                [ao_value[6], ao_value[8], ao_value[9]],
            ]
        )

        for k, ia in enumerate(atmlst):
            ao0, ao1 = mol.aoslice_by_atom()[ia, 2:]
            print(
                f"size of ao_array: {ao_array.shape} elements, size of ao_mat: {ao_mat.shape} elements, size of ao_value: {ao_value.shape} elements, size of grad2force: {grad2force.shape} elements, size of wv: {wv.shape} elements",
                flush=True,
            )
            grad2force_part = np.einsum(
                "inp,xpu,npv,uv->ipx",
                wv,
                ao_value[1:4, :, ao0:ao1],
                ao_array,
                dm1_dft[ao0:ao1],
                optimize=True,
            ) + np.einsum(
                "inp,nxpu,pv,uv->ipx",
                wv,
                ao_mat[:, :, :, ao0:ao1],
                ao_value[0],
                dm1_dft[ao0:ao1],
                optimize=True,
            )
            grad2force_part = -2 * grad2force_part
            grad2force[k, :, p0:p1] = np.reshape(
                grad2force_part,
                (grids.input_level, p1 - p0, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
            )
        print(
            f"current p0: {p0}, p1: {p1}, current size: {lib.current_memory()[0] / 1024:.2f} GB, max size: {max_memory / 1024:.2f} GB",
            flush=True,
        )
        print(
            f"size of ao_array: {ao_array.shape} elements, size of ao_mat: {ao_mat.shape} elements, size of ao_value: {ao_value.shape} elements, size of grad2force: {grad2force.shape} elements",
            flush=True,
        )

    data_dict["rho_cube_dft"] = rho_cube_dft.reshape(
        len(grids.coords), grids.input_level, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
    )
    data_dict["grad2force"] = grad2force
    data_dict["cube_coor"] = gridcube_coords.transpose(0, 2, 1).reshape(
        len(grids.coords), 3, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
    )
    data_dict["ao0_coords"] = ao0_coords.reshape(
        len(grids.coords), EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, dm1_dft.shape[-1]
    )
    data_dict["ao4_coords"] = ao4_coords.reshape(
        grids.input_level,
        len(grids.coords),
        EDGE_SIZE,
        EDGE_SIZE,
        EDGE_SIZE,
        dm1_dft.shape[-1],
    )

    # Test force
    grad_mat = np.zeros(
        (grids.input_level, len(grids.coords), EDGE_SIZE, EDGE_SIZE, EDGE_SIZE)
    )
    grad_mat[0, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.08
    grad_mat[1, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.19
    grad_mat[2, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.72
    grad_mat[3, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.81
    force = np.einsum(
        "p,ipabc,tipabcx->tx",
        grids.weights,
        grad_mat,
        data_dict["grad2force"],
        optimize=True,
    )
    print(
        "Error force DFT: ",
        np.linalg.norm(force - (grad_dft - grad_dft_zeros)),
    )

    vxc_test = np.einsum(
        "g,igabcu,igabc,gabcv->uv",
        grids.weights,
        data_dict["ao4_coords"],
        grad_mat,
        data_dict["ao0_coords"],
        optimize=True,
    )
    vxc_test = lib.hermi_sum(vxc_test, axes=(0, 2, 1))
    vj, vk = mdft.get_jk(mol, dm1_dft)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mdft.xc, spin=mol.spin)
    vk *= hyb
    vxc_test += vj - vk * 0.5
    print(
        hyb, "Error vxc DFT: ", np.linalg.norm(vxc_test - mdft.get_veff(mol, dm1_dft))
    )

    h1e = mdft.get_hcore()
    s1e = mdft.get_ovlp()
    mo_energy, mo_coeff = mdft.eig(vxc_test + h1e, s1e)
    mo_occ = mdft.get_occ(mo_energy, mo_coeff)
    dm_test = mdft.make_rdm1(mo_coeff, mo_occ)

    print("Test MO coefficient difference: ", np.linalg.norm(mo_coeff - mdft.mo_coeff))
    print("Test MO energy difference: ", np.linalg.norm(mo_energy - mdft.mo_energy))
    print(
        "Test dipole difference: ",
        np.linalg.norm(mdft.dip_moment(mol, dm_test) - mdft.dip_moment(mol, dm1_dft)),
    )

    print("Test dm difference: ", np.linalg.norm(dm_test - dm1_dft))
    if is_hermitian(dm_test) and is_hermitian(dm1_dft):
        eigval1, eigvector1 = np.linalg.eigh(dm_test)
        eigval2, eigvector2 = np.linalg.eigh(dm1_dft)
    else:
        eigval1, eigvector1 = np.linalg.eig(dm_test)
        eigval2, eigvector2 = np.linalg.eig(dm1_dft)
    print("Test eigval difference: ", np.linalg.norm(eigval1 - eigval2))
    print(f"convert matrix: {np.linalg.norm(np.linalg.inv(eigvector1) - eigvector1.T)}")
    print(f"convert matrix: {np.linalg.norm(np.linalg.inv(eigvector2) - eigvector2.T)}")


def zmp(dm1_dft, dm1_cc):
    return


def cc(
    mol: pyscf.gto.Mole,
    grids,
    name,
    args,
    evaluate=False,
):
    """
    Generate data for the CCSD method. (Restrict scenario to spin = 0).
    """

    print(f"Generate data for {name}")
    # RHF calculation
    mf = pyscf.scf.RHF(mol)
    mf.max_cycle = 2500
    mf.verbose = 4
    mf.kernel()
    if args.check_convergence and not mf.converged:
        pyscf.scf.addons.dynamic_level_shift_(mf, factor=0.5)
        mf.kernel()
    dm1_hf = mf.make_rdm1(ao_repr=True)
    e_hf = mf.e_tot

    # DFT calculation
    mdft = pyscf.scf.RKS(mol)
    mdft.verbose = 4
    mdft.max_cycle = 200
    mdft.xc = "b3lyp"
    mdft.kernel(mf.make_rdm1())
    if args.check_convergence and not mdft.converged:
        raise ValueError("RKS not converged.")
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    e_dft = mdft.e_tot

    mdft_d3bj = pyscf.scf.RKS(mol)
    mdft_d3bj.verbose = 4
    mdft_d3bj.max_cycle = 200
    mdft_d3bj.xc = "b3lyp-d3bj"
    mdft_d3bj.kernel(dm1_dft)
    e_dft_d3bj = mdft_d3bj.e_tot
    print(f"DFT-D3BJ correct energy: {e_dft_d3bj - e_dft}")

    # CCSD calculation
    mycc = pyscf.cc.CCSD(mf)
    mycc.verbose = 9  # to trace the usage of memory.
    _, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    e_cc = mycc.e_tot + e3ref
    print(f"CCSD(T) energy: {e_cc}")
    energy_train = e_cc - e_dft

    if evaluate:
        dm1_cc = None
        dm1_cc_mo = None
        dm2_cc = None
        grad_cc = None
        del t1, t2, eris, mycc
        gc.collect()
    else:
        l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
        (doo, dov, dvo, dvv), goo, gvv = _gamma1_intermediates(
            mycc, t1, t2, l1, l2, eris
        )
        doo_grad, dvv_grad = doo.copy(), dvv.copy()
        nocc, nvir = t1.shape
        doo[np.diag_indices(nocc)] -= goo.diagonal() * 0.5
        dvv[np.diag_indices(nvir)] += gvv.diagonal() * 0.5
        d1 = (doo, dov, dvo, dvv)
        d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv=False)
        # CC gradient
        if mol.natm == 1:
            grad_cc = np.zeros((mol.natm, 3))  # Fallback to zero gradients
        elif mol.nelectron == 1:
            ghf = pyscf.grad.rhf.Gradients(mf)
            grad_cc = ghf.kernel()
        else:
            # gcc = ccsd_t_grad.Gradients(mycc)
            # grad_cc = gcc.kernel(t1, t2, l1, l2, eris=eris)

            doo_grad -= goo * 0.5
            dvv_grad += gvv * 0.5
            d1_grad = (doo_grad, dov, dvo, dvv_grad)

            dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
            nvir = mycc.nmo - mycc.nocc
            idx = np.tril_indices(nvir)
            vidx = idx[0] * nvir + idx[1]
            dvvvv = dvvvv + dvvvv.transpose(1, 0, 2, 3)
            dvvvv = dvvvv + dvvvv.transpose(0, 1, 3, 2)
            dvvvv = lib.take_2d(dvvvv.reshape(nvir**2, nvir**2), vidx, vidx)
            dvvvv *= 0.25
            d2_grad = (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

            cc_grad = ccsd_grad.Gradients(mycc)
            de = ccsd_grad.grad_elec(
                cc_grad, t1, t2, l1, l2, eris, d1=d1_grad, d2=d2_grad
            )
            cc_grad.de = de + cc_grad.grad_nuc(atmlst=cc_grad.atmlst)
            if cc_grad.mol.symmetry:
                cc_grad.de = cc_grad.symmetrize(cc_grad.de, cc_grad.atmlst)
            cc_grad._finalize()
            grad_cc = cc_grad.de

        del t1, t2, l1, l2
        gc.collect()

        dm1_cc_mo = ccsd_rdm._make_rdm1(mycc, d1, with_frozen=True, ao_repr=False)
        mo = mycc.mo_coeff
        dm1_cc = np.einsum("pi,ij,qj->pq", mo, dm1_cc_mo, mo.conj())
        dm2_cc = ccsd_rdm._make_rdm2(
            mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=True
        )
        del d1, d2, eris, mycc
        gc.collect()

        # Compare CCSD and DFT
        print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} (CCSD vs DFT)")
        cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
        dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
        print(f"{np.linalg.norm(cc_dipole - dft_dipole)} (CCSD vs DFT)")

    data_dict = {}
    # Generate input data
    get_dft_grad(mol, grids, dm1_dft, data_dict)

    if "grad2force" in data_dict and grad_cc is not None:
        # HF gradient
        ghf_hf = pyscf.grad.rhf.Gradients(mf)
        grad_hf = ghf_hf.kernel()

        # DFT gradient
        gdft = mdft.Gradients()
        grad_dft = gdft.kernel()

        gdft_d3bj = mdft_d3bj.Gradients()
        grad_dft_d3bj = gdft_d3bj.kernel()

        data_dict["grad_cc_train"] = grad_cc - grad_dft
        data_dict["grad_hf"] = grad_hf
        data_dict["grad_cc"] = grad_cc
        data_dict["grad_dft"] = grad_dft
        data_dict["grad_dft_d3bj"] = grad_dft_d3bj

    # Calculate the (exchange-correlation energy - DFT energy) on the grids and the grad to force matrix
    data_append_dict = get_dft_energy(
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
    data_dict.update(data_append_dict)

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
            "weights": grids.weights,
        }
    )
    np.savez_compressed(DATA_PATH / f"data_{name}.npz", **data_dict)
