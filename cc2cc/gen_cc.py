# pylint: disable=W0212
import gc

import numpy as np

import pyscf
from pyscf import lib

import pyscf.cc
import pyscf.grad
from pyscf.cc import ccsd_t_lambda
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_rdm
from pyscf.grad import ccsd as ccsd_grad

# from pyscf.cc.ccsd_t_rdm import _gamma1_intermediates
from pyscf.cc.ccsd_t_rdm import _gamma2_intermediates

from cc2cc.utils import diff_rho
from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils.pyscf_ccsd_t_rdm import _gamma1_intermediates
from cc2cc.utils.get_dft_energy_rks import get_cc_energy, get_dft_energy, get_hf_energy
from cc2cc.utils.get_dft_grad_rks import get_dft_grad, get_dft_input
from cc2cc.utils.get_zmp import get_zmp_rks


def is_hermitian(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.conj().T, atol=tol)


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
    mf.conv_tol = 1e-10
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
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.max_cycle = 200
    mycc.BLKMIN = 1
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
        print("Grad done.", flush=True)

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

    mzmp, dm1_zmp = get_zmp_rks(mol, dm1_cc, dm1_dft, grids, max_l=20)

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
        data_append_dict = get_dft_grad(mol, grids, dm1_dft, data_dict)
        data_dict.update(data_append_dict)
        data_append_dict = get_dft_grad(mol, grids, dm1_zmp, data_dict)
        for key in data_append_dict:
            if "dft" in key:
                key_zmp = key.replace("dft", "zmp")
                data_dict[key_zmp] = data_append_dict[key]
            else:
                key_zmp = key + "_zmp"
                data_dict[key_zmp] = data_append_dict[key]

        # HF gradient
        ghf_hf = pyscf.grad.rhf.Gradients(mf)
        grad_hf = ghf_hf.kernel()

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
        data_append_dict = get_dft_input(mol, grids, dm1_dft, data_dict)
        data_dict.update(data_append_dict)
        data_append_dict = get_dft_input(mol, grids, dm1_zmp, data_dict)
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
            "dm1_zmp": dm1_zmp,
            "e_dft_d3bj": e_dft_d3bj,
            "energy_train": energy_train,
            "weights": grids.weights,
        }
    )
    np.savez_compressed(DATA_PATH / f"data_{name}.npz", **data_dict)
