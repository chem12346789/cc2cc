"""Lazy exports for shared helper modules."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "chemistry": ("chemistry", None),
    "runtime": ("runtime", None),
    "units": ("units", None),
    "workflow": ("workflow", None),
    "AU2DEBYE": ("units", "AU2DEBYE"),
    "AU2KCALMOL": ("units", "AU2KCALMOL"),
    "addon_basis": ("chemistry.addon_basis", "addon_basis"),
    "rotate": ("chemistry.rotate", "rotate"),
    "DataRecord": ("workflow.DataRecord", "DataRecord"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
