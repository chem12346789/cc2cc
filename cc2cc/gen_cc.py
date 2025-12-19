# pylint: disable=W0212
import gc
from itertools import product

import numpy as np
import opt_einsum as oe
import torch

import pyscf
from pyscf.cc import ccsd_t_lambda
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_rdm
from pyscf.cc.ccsd_t_rdm import _gamma1_intermediates
from pyscf.cc.ccsd_t_rdm import _gamma2_intermediates
from pyscf.grad import ccsd_t as ccsd_t_grad

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils.modelscf_rks import get_veff_grad_modified_zeros


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

    rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
    rho_hf = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_hf, xctype="GGA")

    if evaluate:
        return rho_dft, {}
    else:
        ni = mdft._numint
        dft_mo_coeff = mdft.mo_coeff
        mf_mo_coeff = mf.mo_coeff

        vxc_lda = ni.eval_xc_eff("LDA,", rho_dft[0], deriv=1, xctype="LDA")[1]
        vxc_vwn = ni.eval_xc_eff(",VWN3", rho_dft[0], deriv=1, xctype="LDA")[1]
        vxc_b88 = ni.eval_xc_eff("B88,", rho_dft, deriv=1, xctype="GGA")[1]
        vxc_lyp = ni.eval_xc_eff(",LYP", rho_dft, deriv=1, xctype="GGA")[1]

        vxc_b3lyp = np.zeros((4, 4, len(grids.coords)))
        vxc_b3lyp[0, 0:1, :] = vxc_lda
        vxc_b3lyp[1, 0:1, :] = vxc_vwn
        vxc_b3lyp[2, :, :] = vxc_b88
        vxc_b3lyp[3, :, :] = vxc_lyp

        wv = grids.weights * vxc_b3lyp
        wv[:, 0, :] *= 0.5

        atmlst = range(mol.natm)
        grad2force = np.zeros((len(atmlst), 4, len(grids.coords), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = mol.aoslice_by_atom()[ia, 2:]
            grad2force[k] = np.einsum(
                "mnp,xpi,npj,ij->mpx",
                wv,
                ao_value[1:4, :, p0:p1],
                ao_array,
                dm1_dft[p0:p1],
                optimize=True,
            ) + np.einsum(
                "mnp,nxpi,pj,ij->mpx",
                wv,
                ao_mat[:, :, :, p0:p1],
                ao_value[0],
                dm1_dft[p0:p1],
                optimize=True,
            )
        grad2force = -grad2force * 2

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

        for i in range(mol.nelec[0]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                dft_mo_coeff[:, i],
                dft_mo_coeff[:, i],
                ao_2_diag,
            )
            kin_dft_grids -= part

        for i in range(mol.nelec[0]):
            part = oe.contract(
                "pm,m,n,pn->p",
                ao_value[0],
                mf_mo_coeff[:, i],
                mf_mo_coeff[:, i],
                ao_2_diag,
            )
            kin_hf_grids -= part

        # nuclear part
        nuc_cc_grids = np.zeros_like(exc_dft_grids)
        nuc_dft_grids = np.zeros_like(exc_dft_grids)
        nuc_hf_grids = np.zeros_like(exc_dft_grids)
        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-5:
                    nuc_cc_grids[i] -= (
                        rho_cc[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_dft_grids[i] -= (
                        rho_dft[0][i] * mol.atom_charges()[i_atom] / distance
                    )
                    nuc_hf_grids[i] -= (
                        rho_hf[0][i] * mol.atom_charges()[i_atom] / distance
                    )

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
                "nuc_cc_grids": nuc_cc_grids,
                "exc_dft_grids": exc_dft_grids,
                "exc_k_dft_grids": exc_k_dft_grids,
                "hatree_dft_grids": hatree_dft_grids,
                "kin_dft_grids": kin_dft_grids,
                "nuc_dft_grids": nuc_dft_grids,
                "exc_k_hf_grids": exc_k_hf_grids,
                "hatree_hf_grids": hatree_hf_grids,
                "kin_hf_grids": kin_hf_grids,
                "nuc_hf_grids": nuc_hf_grids,
                "tol_delta_grids": tol_delta_grids,
                "grad2force": grad2force,
            },
        )


def cc(mol, grids, name, args, evaluate=False):
    """
    Generate data for the CCSD method. (Restrict scenario to spin 0).
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
    # get_veff_modified_rks(mdft, modeldict, lambda_rho=1, dm_tar=dm1_cc)
    mdft.kernel(mf.make_rdm1())
    if args.check_convergence and not mdft.converged:
        raise ValueError("RKS not converged.")
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    e_dft = mdft.e_tot

    # CCSD calculation
    mycc = pyscf.cc.CCSD(mf)
    mycc.verbose = 4
    mycc.direct = True
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
        d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
        # CC gradient
        if mol.natm == 1:
            grad_cc = np.zeros((mol.natm, 3))  # Fallback to zero gradients
        elif mol.nelectron == 1:
            ghf = pyscf.grad.rhf.Gradients(mf)
            grad_cc = ghf.kernel()
        else:
            gcc = ccsd_t_grad.Gradients(mycc)
            grad_cc = gcc.kernel(t1, t2, l1, l2, eris=eris)
        del t1, t2, l1, l2
        gc.collect()

        dm1_cc_mo = ccsd_rdm._make_rdm1(mycc, d1, with_frozen=True, ao_repr=False)
        mo = mycc.mo_coeff
        dm1_cc = np.einsum("pi,ij,qj->pq", mo, dm1_cc_mo, mo.conj())
        dm2_cc = ccsd_rdm._make_rdm2(
            mycc, d1, d2, with_dm1=True, with_dm1=True, ao_repr=True
        )
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

    if "grad2force" in data_dict and grad_cc is not None:
        # HF gradient
        ghf_hf = pyscf.grad.rhf.Gradients(mf)
        grad_hf = ghf_hf.kernel()

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
    rho_cube_dft = grids.gen_cube_rho_rks(rho_dft, mdft._numint, dm1_dft)

    data_dict.update(
        {
            "mol": mol.tostring(format="xyz"),
            "charge": mol.charge,
            "spin": mol.spin,
            "e_cc": e_cc,
            "e_dft": e_dft,
            "e_hf": e_hf,
            "energy_train": energy_train,
            "rho_cube_dft": rho_cube_dft,
            "weights": grids.weights,
        }
    )
    np.savez_compressed(DATA_PATH / f"data_{name}.npz", **data_dict)
