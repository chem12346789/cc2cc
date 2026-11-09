"""Lazy exports for rotated CPU and GPU grid utilities."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "Grids_rotate": ("Grids_rotate", None),
    "GridCPU": ("Grids_rotate", "GridCPU"),
}

_GPU_EXPORTS: dict[str, tuple[str, str | None]] = {
    "GridsGPU_rotate": ("GridsGPU_rotate", None),
    "GridGPU": ("GridsGPU_rotate", "GridGPU"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS, _GPU_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_GPU_EXPORTS))
