"""Density comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

import pyscf
import pyscf.dft


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
