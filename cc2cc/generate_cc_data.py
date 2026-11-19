# pylint: disable=W0212
import gc
import os
import tempfile

import numpy as np

import pyscf
import pyscf.dft

import pyscf.cc
import pyscf.grad
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_rdm
from pyscf.cc import uccsd_t
from pyscf.cc import uccsd_rdm

from pyscf.cc.ccsd_t_rdm import _gamma2_intermediates
from cc2cc.utils.ccsd_t.pyscf_uccsd_t_u_gamma2_intermediates import (
    u_gamma2_intermediates,
)

from cc2cc.utils.helper.chemistry.density_metrics import diff_rho
from cc2cc.utils.helper.runtime.env_var import DATA_PATH
from cc2cc.utils.helper.units import AU2KCALMOL
from cc2cc.utils.reference.reference_cc import (
    make_ccsd,
    ccsd_t_target_density,
    ccsd_t_target_gradient,
    get_cc_energy,
)
from cc2cc.utils.reference.reference_hf import get_hf_energy
from cc2cc.utils.reference.reference_b3lyp import (
    get_dft_energy,
    get_dft_grad,
    get_dft_input,
)
from cc2cc.utils.reference.reference_zmp import get_zmp, zmp_baseline_gradient
from cc2cc.utils.zmp.zmp_constraints import get_zmp_constraint_data
from cc2cc.utils.reference.reference_vv10 import get_vv10_data


def cc(
    mol: pyscf.gto.Mole,
    grids,
    name,
    args,
    evaluate=False,
    *,
    unrestricted=False,
):
    """Generate CCSD(T) reference data with the explicitly selected spin formalism.

    Evaluation and training archive fields are listed in the README.
    """
    if not unrestricted and mol.spin != 0:
        raise ValueError(
            "Restricted CCSD(T) requires mol.spin == 0; set unrestricted=True."
        )
    data_path = DATA_PATH / f"data_{name}.npz"
    data = None
    if getattr(args, "if_continue", False) and data_path.exists():
        with np.load(data_path, allow_pickle=False) as archive:
            if evaluate or archive.get("zmp_complete", True):
                print(f"SKIP: {name} already exists.")
                return
            data = dict(archive)
    if evaluate:
        _generate_evaluation_data(mol, grids, name, args, unrestricted=unrestricted)
    else:
        if data is not None:
            complete_zmp_data(
                mol,
                grids,
                name,
                args,
                data,
                unrestricted=unrestricted,
            )
            return
        _generate_training_data(mol, grids, name, args, unrestricted=unrestricted)


def _hf_reference(mol, name, args, *, unrestricted):
    hf_class = pyscf.scf.UHF if unrestricted else pyscf.scf.RHF
    print(f"Generate data for {name}, spin {mol.spin}")

    mf = hf_class(mol)
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
    return mf, dm1_hf, e_hf


def _cc_reference(mf, *, unrestricted):
    triples = uccsd_t if unrestricted else ccsd_t
    mycc = make_ccsd(mf, unrestricted=unrestricted)
    mycc.verbose = 4
    # To trace the usage of memory, set verbose as 9.
    _, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = triples.kernel(mycc, eris, t1, t2)
    e_cc = mycc.e_tot + e3ref
    print(
        f"{'UCCSD(T)' if unrestricted else 'CCSD(T)'} energy (Hartree): {e_cc}",
        flush=unrestricted,
    )
    return mycc, t1, t2, eris, e_cc


def _dft_reference_data(
    mol,
    grids,
    mf,
    args,
    *,
    evaluate,
    unrestricted,
):
    dft_class = pyscf.dft.uks.UKS if unrestricted else pyscf.dft.rks.RKS
    d3bj_class = pyscf.scf.UKS if unrestricted else pyscf.scf.RKS
    mdft = dft_class(mol, xc="b3lyp")
    mdft.conv_tol = 1e-12
    mdft.conv_tol_grad = 1e-8
    mdft.max_cycle = 200
    mdft.verbose = 4
    mdft.kernel(mf.make_rdm1(ao_repr=True))
    if args.check_convergence and not mdft.converged:
        raise ValueError(f"{'UKS' if unrestricted else 'RKS'} not converged.")
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    e_dft = mdft.e_tot

    mdft_d3bj = d3bj_class(mol)
    mdft_d3bj.verbose = 4
    mdft_d3bj.max_cycle = 200
    mdft_d3bj.xc = "b3lyp-d3bj"
    mdft_d3bj.kernel(dm1_dft)
    e_dft_d3bj = mdft_d3bj.e_tot
    print(f"DFT-D3BJ correction energy (Hartree): {e_dft_d3bj - e_dft}")

    data = get_dft_energy(
        mol,
        grids,
        mdft.mo_coeff,
        dm1_dft,
        evaluate=evaluate,
        unrestricted=unrestricted,
    )
    data.update(
        {
            "mol": mol.tostring(format="xyz"),
            "charge": mol.charge,
            "spin": mol.spin,
            "unrestricted": unrestricted,
            "e_dft": e_dft,
            "e_dft_d3bj": e_dft_d3bj,
            "dm1_dft": dm1_dft,
            "weights": grids.weights,
            "coords": grids.coords,
            "grid_atom_indices": grids.atm_idx,
        }
    )
    return data, mdft, mdft_d3bj, dm1_dft


def _generate_evaluation_data(mol, grids, name, args, *, unrestricted):
    mf, dm1_hf, e_hf = _hf_reference(mol, name, args, unrestricted=unrestricted)
    mycc, t1, t2, eris, e_cc = _cc_reference(mf, unrestricted=unrestricted)
    del t1, t2, eris, mycc
    gc.collect()
    data, _, _, dm1_dft = _dft_reference_data(
        mol,
        grids,
        mf,
        args,
        evaluate=True,
        unrestricted=unrestricted,
    )
    data.update(
        {
            "e_hf": e_hf,
            "e_cc": e_cc,
            "dm1_hf": dm1_hf,
            "dm1_cc": None,
            "energy_train": e_cc - data["e_dft"],
        }
    )
    data.update(get_dft_input(mol, grids, dm1_dft, unrestricted=unrestricted))
    data.update(get_vv10_data(mol, grids, dm1_dft, data["e_dft"], None, args))
    _save_data(DATA_PATH / f"data_{name}.npz", data)


def _generate_training_data(mol, grids, name, args, *, unrestricted):
    gamma2_intermediates = (
        u_gamma2_intermediates if unrestricted else _gamma2_intermediates
    )
    rdm = uccsd_rdm if unrestricted else ccsd_rdm
    hf_grad = pyscf.grad.uhf.Gradients if unrestricted else pyscf.grad.rhf.Gradients
    mf, dm1_hf, e_hf = _hf_reference(mol, name, args, unrestricted=unrestricted)
    mycc, t1, t2, eris, e_cc = _cc_reference(mf, unrestricted=unrestricted)
    l1, l2, d1, d1_grad, dm1_cc_mo, dm1_cc = ccsd_t_target_density(
        mycc, eris, unrestricted=unrestricted
    )
    d2 = gamma2_intermediates(mycc, t1, t2, l1, l2, eris, compress_vvvv=False)
    print("gamma2_intermediates DONE", flush=True)
    grad_cc = ccsd_t_target_gradient(
        mycc, eris, l1, l2, d1_grad, d2, unrestricted=unrestricted
    )

    del t1, t2, l1, l2
    gc.collect()

    print("DM1 done.", flush=True)
    dm2_cc = rdm._make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=True)
    print("DM2 done.", flush=True)
    dm1_cc_mo = np.array(dm1_cc_mo)
    dm2_cc = np.array(dm2_cc)
    del d1, d2, eris, mycc
    gc.collect()

    data_dict = get_cc_energy(
        mol,
        grids,
        mf,
        dm1_cc,
        dm1_cc_mo,
        dm2_cc,
        unrestricted=unrestricted,
    )
    del dm1_cc_mo, dm2_cc
    gc.collect()

    reference, mdft, mdft_d3bj, dm1_dft = _dft_reference_data(
        mol,
        grids,
        mf,
        args,
        evaluate=False,
        unrestricted=unrestricted,
    )
    data_dict.update(reference)
    data_dict.update(get_hf_energy(mol, grids, mf, dm1_hf, unrestricted=unrestricted))
    data_dict.update(
        {
            "e_hf": e_hf,
            "e_cc": e_cc,
            "dm1_hf": dm1_hf,
            "dm1_cc": dm1_cc,
            "energy_train": e_cc - data_dict["e_dft"],
        }
    )
    energy_train = data_dict["energy_train"]
    print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} electrons " "(CCSD vs DFT)")
    cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
    dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
    print(f"{np.linalg.norm(cc_dipole - dft_dipole)} A.U. (CCSD vs DFT)")

    if "tol_cc_grids" in data_dict and "tol_dft_grids" in data_dict:
        data_dict["tol_delta_grids"] = (
            data_dict["tol_cc_grids"] - data_dict["tol_dft_grids"]
        )
    data_append_dict = get_dft_grad(mol, grids, dm1_dft, unrestricted=unrestricted)
    data_dict.update(data_append_dict)

    # HF gradient
    ghf_hf = hf_grad(mf)
    grad_hf = ghf_hf.kernel()

    # DFT gradient
    gdft = mdft.Gradients()
    gdft.grid_response = True
    grad_dft = gdft.kernel()

    gdft_d3bj = mdft_d3bj.Gradients()
    gdft_d3bj.grid_response = True
    grad_dft_d3bj = gdft_d3bj.kernel()

    data_dict["grad_cc_train"] = grad_cc - grad_dft
    data_dict["grad_hf"] = grad_hf
    data_dict["grad_cc"] = grad_cc
    data_dict["grad_dft"] = grad_dft
    data_dict["grad_dft_d3bj"] = grad_dft_d3bj

    if "tol_delta_grids" in data_dict:
        error = np.sum(data_dict["tol_delta_grids"] * grids.weights) - energy_train
        print(f"Error (kcal/mol): {AU2KCALMOL * error}")
        error_hf = np.sum(
            (
                (
                    data_dict["exc_cc_grids"]
                    + data_dict["hatree_cc_grids"]
                    + data_dict["kin_cc_grids"]
                    + data_dict["nuc_cc_grids"]
                    + data_dict["ecp_cc_grids"]
                )
                - (
                    data_dict["exc_k_hf_grids"]
                    + data_dict["hatree_hf_grids"]
                    + data_dict["kin_hf_grids"]
                    + data_dict["nuc_hf_grids"]
                    + data_dict["ecp_hf_grids"]
                )
            )
            * grids.weights
        ) - (e_cc - e_hf)
        print(f"Error HF part (kcal/mol): {AU2KCALMOL * error_hf}")

        error_exc_kinl = (
            np.sum(
                (
                    (
                        data_dict["exc_cc_grids"]
                        + data_dict["hatree_cc_grids"]
                        + data_dict["kinl_cc_grids"]
                        + data_dict["nuc_cc_grids"]
                        + data_dict["ecp_cc_grids"]
                    )
                    - (
                        data_dict["exc_dft_grids"]
                        + data_dict["exc_k_dft_grids"]
                        + data_dict["hatree_dft_grids"]
                        + data_dict["kinl_dft_grids"]
                        + data_dict["nuc_dft_grids"]
                        + data_dict["ecp_dft_grids"]
                    )
                )
                * grids.weights
            )
            - energy_train
        )
        print(f"Error exc_kinl part (kcal/mol): {AU2KCALMOL * error_exc_kinl}")

    data_dict["zmp_complete"] = False
    _save_data(DATA_PATH / f"data_{name}.npz", data_dict)
    print(f"CACHED: {name} CC/DFT references; ZMP is pending.", flush=True)
    complete_zmp_data(
        mol,
        grids,
        name,
        args,
        data_dict,
        unrestricted=unrestricted,
    )


def _save_data(path, data):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez(handle, **data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def complete_zmp_data(mol, grids, name, args, data, *, unrestricted):
    """Append ZMP data without discarding the cached CC/DFT references on failure."""
    if (
        str(data["mol"]) != mol.tostring(format="xyz")
        or int(data["charge"]) != mol.charge
        or int(data["spin"]) != mol.spin
        or bool(data["unrestricted"]) != unrestricted
    ):
        raise ValueError(
            f"{name}: cached CC reference molecule or spin treatment differs."
        )
    for key in ("coords", "weights"):
        if data[key].shape != getattr(grids, key).shape or not np.allclose(
            data[key], getattr(grids, key), rtol=0, atol=1e-12
        ):
            raise ValueError(f"{name}: cached CC reference grid differs.")

    print(f"Using cached CC/DFT references for {name}; completing ZMP.", flush=True)
    dm1_cc, dm1_dft = data["dm1_cc"], data["dm1_dft"]
    data.update(get_vv10_data(mol, grids, dm1_dft, data["e_dft"], dm1_cc, args))
    mzmp, dm1_zmp = get_zmp(mol, dm1_cc, dm1_dft, grids, max_l=20)
    dft_class = pyscf.dft.UKS if unrestricted else pyscf.dft.RKS
    # Match the exact Coulomb and exchange integrals used by the grid labels.
    mdft = dft_class(mol, xc="b3lyp")
    # Reproduce the original SCF grid pruning from its HF initial guess.
    mdft.initialize_grids(mol, data["dm1_hf"])
    e_zmp = mdft.energy_tot(dm1_zmp)
    zmp_energy = get_dft_energy(
        mol,
        grids,
        mzmp.mo_coeff,
        dm1_zmp,
        unrestricted=unrestricted,
    )
    data.update({key.replace("dft", "zmp"): value for key, value in zmp_energy.items()})
    zmp_mapping = get_dft_grad(mol, grids, dm1_zmp, unrestricted=unrestricted)
    data.update(
        {
            key.replace("dft", "zmp") if "dft" in key else key + "_zmp": value
            for key, value in zmp_mapping.items()
        }
    )
    data["tol_delta_zmp_grids"] = data["tol_cc_grids"] - data["tol_zmp_grids"]
    data["grad_zmp"] = zmp_baseline_gradient(mdft, dm1_zmp)
    data["e_zmp"] = e_zmp
    data["dm1_zmp"] = dm1_zmp
    data["energy_train"] = data["e_cc"] - e_zmp
    error_zmp = (
        np.sum(data["tol_delta_zmp_grids"] * grids.weights) - data["energy_train"]
    )
    print(f"Error ZMP (kcal/mol): {AU2KCALMOL * error_zmp}")
    data.update(get_vv10_data(mol, grids, dm1_zmp, e_zmp, dm1_cc, args, suffix="_zmp"))
    data.update(
        get_zmp_constraint_data(
            mdft,
            grids,
            dm1_zmp,
            dm1_cc,
            DATA_PATH / f"data_{name}_zmp",
            args,
        )
    )
    data["zmp_complete"] = True
    _save_data(DATA_PATH / f"data_{name}.npz", data)
