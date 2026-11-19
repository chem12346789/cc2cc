import os
import tempfile

import numpy as np

import pyscf
import pyscf.dft

from cc2cc.utils.helper.chemistry.density_metrics import diff_rho
from cc2cc.utils.helper.runtime.env_var import DATA_PATH
from cc2cc.utils.helper.units import AU2KCALMOL
from cc2cc.utils.reference.reference_cc import cc_reference
from cc2cc.utils.reference.reference_hf import hf_reference
from cc2cc.utils.reference.reference_b3lyp import (
    dft_reference,
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
    if getattr(args, "if_continue", False):
        raise NotImplementedError("B3LYP data continuation is not implemented.")
    if evaluate:
        _generate_evaluation_data(mol, grids, name, args, unrestricted=unrestricted)
    else:
        data_path = DATA_PATH / f"data_{name}.npz"
        cached_data = None
        if data_path.exists():
            with np.load(data_path, allow_pickle=False) as archive:
                cached_data = dict(archive)
        _generate_training_data(
            mol, grids, name, args, unrestricted=unrestricted, cached_data=cached_data
        )


def _generate_evaluation_data(mol, grids, name, args, *, unrestricted):
    data, mf = hf_reference(
        mol, grids, name, args, evaluate=True, unrestricted=unrestricted
    )
    data.update(cc_reference(mol, grids, mf, evaluate=True, unrestricted=unrestricted))
    reference = dft_reference(
        mol,
        grids,
        mf,
        args,
        evaluate=True,
        unrestricted=unrestricted,
    )
    data.update(reference)
    data["energy_train"] = data["e_cc"] - data["e_dft"]
    data.update(
        get_dft_input(mol, grids, reference["dm1_dft"], unrestricted=unrestricted)
    )
    data.update(
        get_vv10_data(mol, grids, reference["dm1_dft"], data["e_dft"], None, args)
    )
    _save_data(DATA_PATH / f"data_{name}.npz", data)


def _generate_training_data(mol, grids, name, args, *, unrestricted, cached_data=None):
    if cached_data is not None:
        complete_zmp_data(
            mol, grids, name, args, cached_data, unrestricted=unrestricted
        )
        return

    data_dict, mf = hf_reference(
        mol, grids, name, args, evaluate=False, unrestricted=unrestricted
    )
    data_dict.update(
        cc_reference(mol, grids, mf, evaluate=False, unrestricted=unrestricted)
    )

    reference = dft_reference(
        mol,
        grids,
        mf,
        args,
        evaluate=False,
        unrestricted=unrestricted,
    )
    data_dict.update(reference)
    data_dict["energy_train"] = data_dict["e_cc"] - data_dict["e_dft"]
    data_dict["grad_cc_train"] = data_dict["grad_cc"] - data_dict["grad_dft"]
    print(
        f"{diff_rho(mol, data_dict['dm1_cc'], data_dict['dm1_dft'], grids):.6f} electrons "
        "(CCSD vs DFT)"
    )
    cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=data_dict["dm1_cc"], unit="A.U.")
    dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=data_dict["dm1_dft"], unit="A.U.")
    print(f"{np.linalg.norm(cc_dipole - dft_dipole)} A.U. (CCSD vs DFT)")

    if "tol_cc_grids" in data_dict and "tol_dft_grids" in data_dict:
        data_dict["tol_delta_grids"] = (
            data_dict["tol_cc_grids"] - data_dict["tol_dft_grids"]
        )

    data_dict["zmp_complete"] = False
    _save_data(DATA_PATH / f"data_{name}.npz", data_dict)
    print(f"CACHED: {name} CC/DFT references; ZMP is pending.", flush=True)
    complete_zmp_data(mol, grids, name, args, data_dict, unrestricted=unrestricted)


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
        mol, grids, mzmp.mo_coeff, dm1_zmp, unrestricted=unrestricted
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
            mdft, grids, dm1_zmp, dm1_cc, DATA_PATH / f"data_{name}_zmp", args
        )
    )
    data["zmp_complete"] = True
    _save_data(DATA_PATH / f"data_{name}.npz", data)
