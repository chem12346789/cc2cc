# pylint: disable=W0212
import gc
from math import erf

import numpy as np
import opt_einsum as oe
import torch

import pyscf
from pyscf import lib

import pyscf.cc
from pyscf.cc import uccsd_t
from pyscf.cc import uccsd_rdm
from pyscf.grad import uccsd as uccsd_grad

from cc2cc.gen_cc import block_loop_rdm2, is_hermitian

from cc2cc.utils.pyscf_uccsd_t_lambda import kernel as uccsd_t_lambda_kernel
from cc2cc.utils.pyscf_uccsd_t_u_gamma1_intermediates import u_gamma1_intermediates
from cc2cc.utils.pyscf_uccsd_t_u_gamma2_intermediates import u_gamma2_intermediates
from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils.modelscf_uks import get_veff_grad_modified_zeros
from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_SIZE


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
        dft_mo_coeff = mdft.mo_coeff
        mf_mo_coeff = mf.mo_coeff

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
    mdft = pyscf.scf.UKS(mol)
    mdft.grids = grids
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

    nao = dm1_dft.shape[-1]
    rho_cube_dft = np.zeros((len(grids.coords), grids.input_level, EDGE_SIZE**3))

    ni = mdft._numint
    step = int(max_memory * 1024**2 / (nao * EDGE_SIZE**3 * 32 * 8))
    # 32 is the number of elements in the ao_array and ao_mat, 8 is the size of float64 in bytes
    print(f"Step size: {step}")
    for p0, p1 in lib.prange(0, len(grids.coords), step):
        if grids.screen_index is None:
            mask = None
        else:
            mask = grids.screen_index[p0:p1]
        coords_ = grids.coords[p0:p1]
        gridcube = grids.gen_cube(mol, dm1_dft, coords_, mask)
        rho_cube_dft_part, wv, ao_value = gridcube.gen_cube_rho_uks(
            ni, dm1_dft, ao_deriv=2, require_vxc=True
        )
        rho_cube_dft[p0:p1] = rho_cube_dft_part

        wv = wv.reshape(gridcube.input_level, 2, 4, len(gridcube.coords))
        wv[:, :, 0, :] *= 0.5

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
                "isnp,xpu,npv,suv->ipx",
                wv,
                ao_value[1:4, :, ao0:ao1],
                ao_array,
                dm1_dft[:, ao0:ao1],
                optimize=True,
            ) + np.einsum(
                "isnp,nxpu,pv,suv->ipx",
                wv,
                ao_mat[:, :, :, ao0:ao1],
                ao_value[0],
                dm1_dft[:, ao0:ao1],
                optimize=True,
            )
            grad2force_part = -grad2force_part * 2
            grad2force[k, :, p0:p1] = np.reshape(
                grad2force_part,
                (grids.input_level, p1 - p0, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
            )
        print(
            f"current p0: {p0}, p1: {p1}, current size: {lib.current_memory()[0] / 1024:.2f} GB, max size: {max_memory / 1024:.2f} GB",
        )
        print(
            f"size of ao_array: {ao_array.shape} elements, size of ao_mat: {ao_mat.shape} elements, size of ao_value: {ao_value.shape} elements, size of grad2force: {grad2force.shape} elements",
            flush=True,
        )

    data_dict["rho_cube_dft"] = rho_cube_dft.reshape(
        len(grids.coords), grids.input_level, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
    )
    data_dict["grad2force"] = grad2force

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
    print("Error force DFT: ", np.linalg.norm(force - (grad_dft - grad_dft_zeros)))
    data_dict["grad_dft_zeros"] = grad_dft_zeros


def uzmp(dm1_dft, dm1_cc):
    return


def ucc(mol, grids, name, args, evaluate=False):
    """
    Generate data for the UCCSD method.
    """
    print(f"Generate data for {name}, spin {mol.spin}")

    # UHF calculation
    mf = pyscf.scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
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
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 200
    _, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = uccsd_t.kernel(mycc, eris, t1, t2)
    e_cc = mycc.e_tot + e3ref
    print(f"UCCSD(T) energy: {e_cc}", flush=True)
    energy_train = e_cc - e_dft

    if evaluate:
        dm1_cc = None
        dm1_cc_mo = None
        dm2_cc = None
        grad_cc = None
        del t1, t2, eris, mycc
        gc.collect()
    else:
        l1, l2 = uccsd_t_lambda_kernel(mycc, eris, t1, t2)[1:]
        print("uccsd_t_lambda DONE", flush=True)
        d1, (goo, gOO, gvv, gVV) = u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        print("u_gamma1_intermediates DONE", flush=True)
        d2 = u_gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
        print("u_gamma2_intermediates DONE", flush=True)
        nocca, noccb, nvira, nvirb = t2[1].shape
        ((doo, dOO), (dov, dOV), (dvo, dVO), (dvv, dVV)) = d1
        doo_grad, dvv_grad = doo.copy(), dvv.copy()
        dOO_grad, dVV_grad = dOO.copy(), dVV.copy()
        doo[np.diag_indices(nocca)] -= goo.diagonal()
        dOO[np.diag_indices(noccb)] -= gOO.diagonal()
        dvv[np.diag_indices(nvira)] += gvv.diagonal()
        dVV[np.diag_indices(nvirb)] += gVV.diagonal()
        d1 = ((doo, dOO), (dov, dOV), (dvo, dVO), (dvv, dVV))

        # CC gradient
        if mol.natm == 1:
            grad_cc = np.zeros((mol.natm, 3))  # Fallback to zero gradients
        elif mol.nelectron == 1:
            ghf = pyscf.grad.uhf.Gradients(mf)
            grad_cc = ghf.kernel()
        else:
            # gcc = uccsd_t_grad.Gradients(mycc)
            # grad_cc = gcc.kernel(t1, t2, l1, l2, eris=eris)

            doo_grad -= goo
            dOO_grad -= gOO
            dvv_grad += gvv
            dVV_grad += gVV
            d1_grad = (
                (doo_grad, dOO_grad),
                (dov, dOV),
                (dvo, dVO),
                (dvv_grad, dVV_grad),
            )
            (
                (dovov, dovOV, dOVov, dOVOV),
                (dvvvv, dvvVV, dVVvv, dVVVV),
                (doooo, dooOO, dOOoo, dOOOO),
                (doovv, dooVV, dOOvv, dOOVV),
                (dovvo, dovVO, dOVvo, dOVVO),
                (dvvov, dvvOV, dVVov, dVVOV),
                (dovvv, dovVV, dOVvv, dOVVV),
                (dooov, dooOV, dOOov, dOOOV),
            ) = d2
            idxa = np.tril_indices(nvira)
            idxa = idxa[0] * nvira + idxa[1]
            idxb = np.tril_indices(nvirb)
            idxb = idxb[0] * nvirb + idxb[1]
            dvvvv = dvvvv + dvvvv.transpose(1, 0, 2, 3)
            dvvvv = lib.take_2d(dvvvv.reshape(nvira**2, nvira**2), idxa, idxa)
            dvvvv *= 0.5
            dvvVV = dvvVV + dvvVV.transpose(1, 0, 2, 3)
            dvvVV = lib.take_2d(dvvVV.reshape(nvira**2, nvirb**2), idxa, idxb)
            dVVVV = dVVVV + dVVVV.transpose(1, 0, 2, 3)
            dVVVV = lib.take_2d(dVVVV.reshape(nvirb**2, nvirb**2), idxb, idxb)
            dVVVV *= 0.5
            d2_grad = (
                (dovov, dovOV, dOVov, dOVOV),
                (dvvvv, dvvVV, dVVvv, dVVVV),
                (doooo, dooOO, dOOoo, dOOOO),
                (doovv, dooVV, dOOvv, dOOVV),
                (dovvo, dovVO, dOVvo, dOVVO),
                (dvvov, dvvOV, dVVov, dVVOV),
                (dovvv, dovVV, dOVvv, dOVVV),
                (dooov, dooOV, dOOov, dOOOV),
            )

            cc_grad = uccsd_grad.Gradients(mycc)
            de = uccsd_grad.grad_elec(
                cc_grad, t1, t2, l1, l2, eris, d1=d1_grad, d2=d2_grad
            )
            cc_grad.de = de + cc_grad.grad_nuc(atmlst=cc_grad.atmlst)
            if cc_grad.mol.symmetry:
                cc_grad.de = cc_grad.symmetrize(cc_grad.de, cc_grad.atmlst)
            cc_grad._finalize()
            grad_cc = cc_grad.de

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

    data_dict = {}
    # Generate input data
    get_dft_grad(mol, grids, dm1_dft, data_dict)

    if "grad2force" in data_dict and grad_cc is not None:
        # HF gradient
        ghf = pyscf.grad.uhf.Gradients(mf)
        grad_hf = ghf.kernel()

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
