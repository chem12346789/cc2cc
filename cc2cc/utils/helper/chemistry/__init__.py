"""Lazy exports for chemistry helpers."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "addon_basis": ("addon_basis", None),
    "basis": ("basis", None),
    "density_metrics": ("density_metrics", None),
    "grid_energy": ("grid_energy", None),
    "rotate": ("rotate", None),
    "tensor_helper": ("tensor_helper", None),
    "def_ecp_list": ("basis", "def_ecp_list"),
    "aug_cc_atom_list": ("basis", "aug_cc_atom_list"),
    "aug_cc_pp_atom_list": ("basis", "aug_cc_pp_atom_list"),
    "special_atom_list": ("basis", "special_atom_list"),
    "cc_atom_list": ("basis", "cc_atom_list"),
    "cc_pp_atom_list": ("basis", "cc_pp_atom_list"),
    "get_ecp": ("basis", "get_ecp"),
    "gen_basis": ("basis", "gen_basis"),
    "gen_ecp": ("basis", "gen_ecp"),
    "diff_rho": ("density_metrics", "diff_rho"),
    "diff_I_value": ("density_metrics", "diff_I_value"),
    "nuclear_energy_density": ("grid_energy", "nuclear_energy_density"),
    "ecp_energy_density": ("grid_energy", "ecp_energy_density"),
    "orbital_gradient_squared": ("grid_energy", "orbital_gradient_squared"),
    "block_loop_rdm2": ("grid_energy", "block_loop_rdm2"),
    "MASS": ("rotate", "MASS"),
    "get_barycenter": ("rotate", "get_barycenter"),
    "get_inertia_moment": ("rotate", "get_inertia_moment"),
    "rotation_matrix_from_vectors": ("rotate", "rotation_matrix_from_vectors"),
    "parallel_einsum": ("tensor_helper", "parallel_einsum"),
    "w_einsum": ("tensor_helper", "w_einsum"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
