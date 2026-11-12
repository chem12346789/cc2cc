"""Lazy exports for workflow helpers."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "DataRecord": ("DataRecord", None),
    "run_helpers": ("run_helpers", None),
    "release_memory": ("run_helpers", "release_memory"),
    "build_record_path": ("run_helpers", "build_record_path"),
    "load_post_scf_density": ("run_helpers", "load_post_scf_density"),
    "load_post_scf_vv10": ("run_helpers", "load_post_scf_vv10"),
    "should_skip": ("run_helpers", "should_skip"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
