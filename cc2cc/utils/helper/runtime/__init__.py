"""Lazy exports for runtime helpers."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "computer_info": ("computer_info", None),
    "env_var": ("env_var", None),
    "timer": ("timer", None),
    "xc_profile": ("xc_profile", None),
    "print_computer_info": ("computer_info", "print_computer_info"),
    "EDGE_SIZE": ("env_var", "EDGE_SIZE"),
    "CUBE_MIDDLE": ("env_var", "CUBE_MIDDLE"),
    "EDGE_LEN": ("env_var", "EDGE_LEN"),
    "TEST": ("env_var", "TEST"),
    "MAIN_PATH": ("env_var", "MAIN_PATH"),
    "DATA_DIR": ("env_var", "DATA_DIR"),
    "DATA_PATH": ("env_var", "DATA_PATH"),
    "DATA_TEST_DIR": ("env_var", "DATA_TEST_DIR"),
    "DATA_TEST_PATH": ("env_var", "DATA_TEST_PATH"),
    "CHECKPOINTS_PATH": ("env_var", "CHECKPOINTS_PATH"),
    "MAX_GPU_JK_NAO": ("env_var", "MAX_GPU_JK_NAO"),
    "Timer": ("timer", "Timer"),
    "XCStageProfiler": ("xc_profile", "XCStageProfiler"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
