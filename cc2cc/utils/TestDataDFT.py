from __future__ import annotations

import os
from pathlib import Path
from timeit import default_timer as timer
import warnings
from typing import Any

import numpy as np

import pyscf
import pyscf.dft

from cc2cc.utils.env_var import DATA_TEST_PATH


def _to_numpy(array: Any) -> np.ndarray:
    """Convert cupy/numpy-like arrays to numpy arrays."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def _prepare_density_eval(
    mol: pyscf.M,
    dm1_compare1: np.ndarray,
    dm1_compare2: np.ndarray,
    grids: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = _to_numpy(grids.coords)
    weights = _to_numpy(grids.weights)
    dm1_compare1 = _to_numpy(dm1_compare1)
    dm1_compare2 = _to_numpy(dm1_compare2)
    if dm1_compare1.ndim != dm1_compare2.ndim:
        raise ValueError("dm1_compare1 and dm1_compare2 must have the same dimension.")
    if dm1_compare1.ndim == 3:
        dm1_compare1 = dm1_compare1[0] + dm1_compare1[1]
        dm1_compare2 = dm1_compare2[0] + dm1_compare2[1]
    ao = pyscf.dft.numint.eval_ao(mol, coords, deriv=0)
    return ao, weights, dm1_compare1, dm1_compare2


def diff_rho(
    mol: pyscf.M,
    dm1_compare1: np.ndarray,
    dm1_compare2: np.ndarray,
    grids: Any,
) -> float:
    """Calculate the integrated density difference."""
    ao, weights, dm1_compare1, dm1_compare2 = _prepare_density_eval(
        mol, dm1_compare1, dm1_compare2, grids
    )
    drho = pyscf.dft.numint.eval_rho(mol, ao, dm1_compare1 - dm1_compare2, xctype="LDA")
    return float(np.sum(np.abs(drho) * weights))


def diff_I_value(
    mol: pyscf.M,
    dm1_compare1: np.ndarray,
    dm1_compare2: np.ndarray,
    grids: Any,
) -> float:
    r"""Compute normalized density mismatch I-value."""
    ao, weights, dm1_compare1, dm1_compare2 = _prepare_density_eval(
        mol, dm1_compare1, dm1_compare2, grids
    )
    rho1 = pyscf.dft.numint.eval_rho(mol, ao, dm1_compare1, xctype="LDA")
    rho2 = pyscf.dft.numint.eval_rho(mol, ao, dm1_compare2, xctype="LDA")
    drho = rho1 - rho2
    numerator = np.sum(np.abs(drho) ** 2 * weights)
    denominator = np.sum(np.abs(rho1) ** 2 * weights) + np.sum(
        np.abs(rho2) ** 2 * weights
    )
    return float(numerator / denominator)


class TestDataDFT:
    """Generate/load cached DFT reference data for a molecule."""

    def __init__(
        self,
        mol: pyscf.M,
        name: str,
        xc_code: str,
        disp: str | None,
    ) -> None:
        self.mol = mol
        xc_code_disp = xc_code if disp is None else f"{xc_code}-{disp}"
        print(f"Testing DFT {xc_code_disp} for {name}")
        path_to_data = DATA_TEST_PATH / f"{name}_cc.npz"

        if (path_to_data).exists():
            print(f"Data for {name} loaded from file.")
            data_frame = dict(np.load(path_to_data, allow_pickle=True).items())
        else:
            data_frame = {"mol_corr": mol.atom_coords()}

        if_update = False
        dm1_dft = data_frame.get("dm1_dft")
        if f"e_dft-{xc_code_disp}" not in data_frame:
            data_frame.update(self._run_ks(dm1_dft, xc_code_disp))
            if_update = True

        mol_corr = data_frame["mol_corr"]
        if np.linalg.norm(mol.atom_coords() - mol_corr, ord=1) > 1e-6:
            print("Molecule coordinates are different.")
            warnings.warn(
                f"Coordinates of {name} are different from the saved data. "
                "Please check the coordinates or regenerate the data."
            )
            data_frame.update(self._run_ks(dm1_dft, xc_code_disp))
            if_update = True

        self.dm1_dft = data_frame["dm1_dft"]
        self.grad_dft = data_frame[f"grad_dft-{xc_code_disp}"]
        self.e_dft = data_frame[f"e_dft-{xc_code_disp}"]
        self.dft_dipole = data_frame[f"dft_dipole-{xc_code_disp}"]

        print(f"Data for {name} loaded.")
        if if_update:
            print(f"Data for {name} saved to file.")
            np.savez(path_to_data, **data_frame)

    def _run_ks(self, dm1_dft: np.ndarray | None, xc_code_disp: str) -> dict[str, Any]:
        if self.mol.spin == 0:
            return self.test_mol_rks(dm1_dft, xc_code_disp)
        return self.test_mol_uks(dm1_dft, xc_code_disp)

    def _test_mol_ks(
        self,
        dm1_dft: np.ndarray | None,
        xc_code_disp: str,
        is_uks: bool,
    ) -> dict[str, Any]:
        ks_class = pyscf.scf.UKS if is_uks else pyscf.scf.RKS
        method_name = "UKS" if is_uks else "RKS"
        time_start = timer()
        mdft = ks_class(self.mol).density_fit()
        mdft.xc = xc_code_disp
        mdft.verbose = 4
        mdft.grids.level = 4
        mdft.level_shift = 0.1
        if dm1_dft is None:
            mdft.kernel()
        else:
            mdft.kernel(dm0=dm1_dft)
        if not mdft.converged:
            raise ValueError(f"{method_name} not converged.")
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        time_dft = timer() - time_start
        data = {
            f"e_dft-{xc_code_disp}": mdft.e_tot,
            f"dft_dipole-{xc_code_disp}": pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=dm1_dft,
                unit="A.U.",
            ),
            f"time_dft-{xc_code_disp}": time_dft,
            f"grad_dft-{xc_code_disp}": mdft.Gradients().kernel(),
        }
        if xc_code_disp == "b3lyp":
            data["dm1_dft"] = dm1_dft
        return data

    def test_mol_rks(
        self, dm1_dft: np.ndarray | None, xc_code_disp: str
    ) -> dict[str, Any]:
        return self._test_mol_ks(dm1_dft, xc_code_disp, is_uks=False)

    def test_mol_uks(
        self, dm1_dft: np.ndarray | None, xc_code_disp: str
    ) -> dict[str, Any]:
        return self._test_mol_ks(dm1_dft, xc_code_disp, is_uks=True)
