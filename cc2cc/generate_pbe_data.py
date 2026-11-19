"""PBE supplements sharing the existing CC and fixed-ZMP reference data."""

import os
import tempfile

import numpy as np
import pyscf.dft

from cc2cc.utils.helper.runtime.env_var import DATA_PATH
from cc2cc.utils.reference.reference_b3lyp import (
    get_dft_energy,
    get_dft_grad,
    get_dft_input,
)
from cc2cc.utils.reference.reference_zmp import zmp_baseline_gradient
from cc2cc.utils.zmp.zmp_constraints import get_zmp_constraint_data


def generate_pbe_data(mol, grids, name, args, *, evaluate=False, unrestricted):
    """Write a separate PBE archive without rerunning CC or ZMP inversion."""
    source_path = DATA_PATH / f"data_{name}.npz"
    output_path = DATA_PATH / f"data_{name}_pbe.npz"
    if args.zmp_response_vv10:
        raise ValueError("PBE supplements do not include VV10.")
    if args.if_continue and output_path.exists():
        print(f"SKIP: {output_path.name} already exists.")
        return
    if not source_path.is_file():
        raise ValueError(f"PBE generation requires the reference archive {source_path}.")

    with np.load(source_path, allow_pickle=False) as reference:
        if (
            str(reference["mol"]) != mol.tostring(format="xyz")
            or int(reference["charge"]) != mol.charge
            or int(reference["spin"]) != mol.spin
        ):
            raise ValueError(f"{name}: molecule differs from the reference archive.")
        for key in ("coords", "weights"):
            saved = reference[key]
            current = getattr(grids, key)
            if saved.shape != current.shape or not np.allclose(
                saved, current, rtol=0, atol=1e-12
            ):
                raise ValueError(f"{name}: saved {key} differ; use the original grid.")

        mf_class = pyscf.dft.UKS if unrestricted else pyscf.dft.RKS
        mf = mf_class(mol, xc="pbe")
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 200
        mf.kernel(reference["dm1_dft"])
        if not mf.converged:
            raise RuntimeError(f"{name}: PBE SCF did not converge.")
        dm_pbe = mf.make_rdm1()
        data = {
            "source_archive": source_path.name,
            "baseline_xc": "pbe",
            "mol": mol.tostring(format="xyz"),
            "basis": args.basis,
            "pbe_mol": mol.dumps(),
            "charge": mol.charge,
            "spin": mol.spin,
            "unrestricted": unrestricted,
            "coords": grids.coords,
            "weights": grids.weights,
            "e_cc": reference["e_cc"],
            "e_dft": mf.e_tot,
            "dm1_dft": dm_pbe,
            "energy_train": reference["e_cc"] - mf.e_tot,
        }
        if evaluate:
            data["rho_cube_dft"] = get_dft_input(
                mol, grids, dm_pbe, unrestricted=unrestricted, ni=mf._numint,
            )["rho_cube_dft"]
        else:
            pbe_grid = get_dft_energy(
                mol, grids, mf.mo_coeff, dm_pbe, unrestricted=unrestricted, xc="pbe",
            )
            data.update(pbe_grid)
            data["tol_delta_grids"] = (
                reference["tol_cc_grids"] - pbe_grid["tol_dft_grids"]
            )
            mappings = get_dft_grad(mol, grids, dm_pbe, unrestricted=unrestricted)
            data.update(mappings)
            grad = mf.nuc_grad_method()
            grad.grid_response = True
            data["grad_dft"] = grad.kernel()
            data["grad_cc"] = reference["grad_cc"]
            data["grad_cc_train"] = data["grad_cc"] - data["grad_dft"]

            # Keep the inversion density fixed to isolate the baseline change.
            dm_zmp = reference["dm1_zmp"]
            overlap = mf.get_ovlp()
            _, coeff_zmp = mf.eig(-overlap @ dm_zmp @ overlap, overlap)
            zmp_grid = get_dft_energy(
                mol, grids, coeff_zmp, dm_zmp, unrestricted=unrestricted, xc="pbe",
            )
            data.update({
                key.replace("_dft", "_zmp"): val for key, val in zmp_grid.items()
            })
            data["e_zmp"] = mf.energy_tot(dm_zmp)
            data["energy_train_zmp"] = reference["e_cc"] - data["e_zmp"]
            data["tol_delta_zmp_grids"] = (
                reference["tol_cc_grids"] - zmp_grid["tol_dft_grids"]
            )
            data["grad_zmp"] = zmp_baseline_gradient(mf, dm_zmp)
            data.update(get_zmp_constraint_data(
                mf, grids, dm_zmp, reference["dm1_cc"],
                DATA_PATH / f"data_{name}_pbe_zmp", args,
            ))

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez(handle, **data)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"SAVED: {output_path.name}")
