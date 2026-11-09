"""Lazy exports for ZMP inversion, constraints, and training."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "zmp": ("zmp", None),
    "zmp_constraints": ("zmp_constraints", None),
    "zmp_ov_training": ("zmp_ov_training", None),
    "CONV_TOL_DM": ("zmp", "CONV_TOL_DM"),
    "CONV_TOL_DIIS": ("zmp", "CONV_TOL_DIIS"),
    "DIIS": ("zmp", "DIIS"),
    "RZMP": ("zmp", "RZMP"),
    "UZMP": ("zmp", "UZMP"),
    "get_zmp_constraint_data": ("zmp_constraints", "get_zmp_constraint_data"),
    "GaussianOrbitalCache": ("zmp_ov_training", "GaussianOrbitalCache"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
