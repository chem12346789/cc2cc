# pylint: disable=W0212
import gc
from math import erf

import numpy as np
import opt_einsum as oe
import torch

import pyscf
from pyscf import lib
import pyscf.cc
import pyscf.grad
from pyscf.cc import uccsd_t
from pyscf.cc import uccsd_rdm
from pyscf.grad import uccsd as uccsd_grad

from cc2cc.utils.pyscf_uccsd_t_lambda import kernel as uccsd_t_lambda_kernel
from cc2cc.utils.pyscf_uccsd_t_u_gamma1_intermediates import u_gamma1_intermediates
from cc2cc.utils.pyscf_uccsd_t_u_gamma2_intermediates import u_gamma2_intermediates

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils.get_dft_energy_uks import get_cc_energy, get_dft_energy, get_hf_energy
from cc2cc.utils.get_dft_grad_uks import get_dft_grad, get_dft_input
from cc2cc.utils.get_zmp import get_zmp_uks


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
    for level_shift in range(5):
        if args.check_convergence and not mf.converged:
            mf.level_shift = 4 ** (level_shift)
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
        (doo, dOO), (dov, dOV), (dvo, dVO), (dvv, dVV) = d1
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
        print("Grad done.", flush=True)

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
        print("DM1 done.", flush=True)
        dm2_cc = uccsd_rdm._make_rdm2(
            mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=True
        )
        print("DM2 done.", flush=True)
        dm1_cc_mo = np.array(dm1_cc_mo)
        dm2_cc = np.array(dm2_cc)
        del d1, d2, eris, mycc
        gc.collect()

        # Compare CCSD and DFT
        print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} (CCSD vs DFT)")
        cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
        dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
        print(f"{np.linalg.norm(cc_dipole - dft_dipole)} (CCSD vs DFT)")

    mzmp, dm1_zmp = get_zmp_uks(mol, dm1_cc, dm1_dft, grids, max_l=20)

    # Generate input data
    data_dict = {}

    # Calculate the (exchange-correlation energy - DFT energy) on the grids and the grad to force matrix
    data_append_dict = get_cc_energy(
        mol,
        grids,
        mf,
        dm1_cc,
        dm1_cc_mo,
        dm2_cc,
        evaluate=evaluate,
    )
    data_dict.update(data_append_dict)
    data_append_dict = get_dft_energy(
        mol,
        grids,
        mdft,
        dm1_dft,
        evaluate=evaluate,
    )
    data_dict.update(data_append_dict)
    data_append_dict = get_hf_energy(
        mol,
        grids,
        mf,
        dm1_hf,
        evaluate=evaluate,
    )
    data_dict.update(data_append_dict)
    if "tol_cc_grids" in data_dict and "tol_dft_grids" in data_dict:
        data_dict["tol_delta_grids"] = (
            data_dict["tol_cc_grids"] - data_dict["tol_dft_grids"]
        )

    data_append_dict = get_dft_energy(
        mol,
        grids,
        mzmp,
        dm1_zmp,
        evaluate=evaluate,
    )
    for key in data_append_dict:
        key_zmp = key.replace("dft", "zmp")
        data_dict[key_zmp] = data_append_dict[key]
    if "tol_cc_grids" in data_dict and "tol_zmp_grids" in data_dict:
        data_dict["tol_delta_zmp_grids"] = (
            data_dict["tol_cc_grids"] - data_dict["tol_zmp_grids"]
        )

    if not evaluate:
        data_append_dict = get_dft_grad(mol, grids, dm1_dft)
        data_dict.update(data_append_dict)
        data_append_dict = get_dft_grad(mol, grids, dm1_zmp)
        for key in data_append_dict:
            if "dft" in key:
                key_zmp = key.replace("dft", "zmp")
                data_dict[key_zmp] = data_append_dict[key]
            else:
                key_zmp = key + "_zmp"
                data_dict[key_zmp] = data_append_dict[key]
        # HF gradient
        ghf = pyscf.grad.uhf.Gradients(mf)
        grad_hf = ghf.kernel()

        # DFT gradient
        gdft = mdft.Gradients()
        grad_dft = gdft.kernel()

        gdft_d3bj = mdft_d3bj.Gradients()
        grad_dft_d3bj = gdft_d3bj.kernel()

        grad_zmp = mzmp.Gradients()
        grad_zmp = grad_zmp.kernel()

        data_dict["grad_cc_train"] = grad_cc - grad_dft
        data_dict["grad_hf"] = grad_hf
        data_dict["grad_cc"] = grad_cc
        data_dict["grad_dft"] = grad_dft
        data_dict["grad_dft_d3bj"] = grad_dft_d3bj
        data_dict["grad_zmp"] = grad_zmp
    else:
        data_append_dict = get_dft_input(mol, grids, dm1_dft)
        data_dict.update(data_append_dict)
        data_append_dict = get_dft_input(mol, grids, dm1_zmp)
        for key in data_append_dict:
            key_zmp = key.replace("dft", "zmp")
            data_dict[key_zmp] = data_append_dict[key]

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

    if "tol_delta_zmp_grids" in data_dict:
        e_zmp = mzmp.energy_tot(dm1_zmp)
        energy_train = e_cc - e_zmp
        error_zmp = (
            np.sum(data_dict["tol_delta_zmp_grids"] * grids.weights) - energy_train
        )
        print(f"Error ZMP: {AU2KCALMOL * error_zmp}")
        data_dict["e_zmp"] = e_zmp
        data_dict["dm1_zmp"] = dm1_zmp

    data_dict.update(
        {
            "mol": mol.tostring(format="xyz"),
            "charge": mol.charge,
            "spin": mol.spin,
            "e_hf": e_hf,
            "e_cc": e_cc,
            "e_dft": e_dft,
            "e_dft_d3bj": e_dft_d3bj,
            "dm1_hf": dm1_hf,
            "dm1_cc": dm1_cc,
            "dm1_dft": dm1_dft,
            "energy_train": energy_train,
            "weights": grids.weights,
        }
    )
    np.savez_compressed(DATA_PATH / f"data_{name}.npz", **data_dict)
