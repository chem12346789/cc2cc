"""Public utility API for cc2cc.

Most CPU exports are loaded eagerly, while SCF helper functions and GPU exports
remain lazy.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

NO_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "add_args": ("cc2cc.utils.parser", "add_args"),
    "config_list": ("cc2cc.utils.parser", "config_list"),
    "process_config": ("cc2cc.utils.parser", "process_config"),
    "gen_mole": ("cc2cc.utils.mol", "gen_mole"),
    "AU2KCALMOL": ("cc2cc.utils.mol", "AU2KCALMOL"),
    "AU2DEBYE": ("cc2cc.utils.mol", "AU2DEBYE"),
    "DATA_PATH": ("cc2cc.utils.env_var", "DATA_PATH"),
    "MAIN_PATH": ("cc2cc.utils.env_var", "MAIN_PATH"),
    "print_computer_info": ("cc2cc.utils.computer_info", "print_computer_info"),
    "TestDataDFT": ("cc2cc.utils.TestDataDFT", "TestDataDFT"),
    "diff_rho": ("cc2cc.utils.TestDataDFT", "diff_rho"),
    "Timer": ("cc2cc.utils.timer", "Timer"),
    "DataRecord": ("cc2cc.utils.DataRecord", "DataRecord"),
    "ModelClass": ("cc2cc.utils.ModelClass", "ModelClass"),
    "Grid": ("cc2cc.utils.Grids", "Grid"),
    "GridCPU": ("cc2cc.utils.Grids", "GridCPU"),
}

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "get_veff_modified_rks": (
        "cc2cc.utils.modelscf_rks",
        "get_veff_modified_rks",
    ),
    "get_veff_grad_modified_rks": (
        "cc2cc.utils.modelscf_rks",
        "get_veff_grad_modified_rks",
    ),
    "get_veff_modified_uks": (
        "cc2cc.utils.modelscf_uks",
        "get_veff_modified_uks",
    ),
    "get_veff_grad_modified_uks": (
        "cc2cc.utils.modelscf_uks",
        "get_veff_grad_modified_uks",
    ),
}

_GPU_EXPORTS: dict[str, tuple[str, str]] = {
    "GridGPU": ("cc2cc.utils.GridsGPU", "GridGPU"),
    "get_veff_modified_rks_gpu": (
        "cc2cc.utils.modelscf_rks_gpu",
        "get_veff_modified_rks_gpu",
    ),
    "get_veff_grad_modified_rks_gpu": (
        "cc2cc.utils.modelscf_rks_gpu",
        "get_veff_grad_modified_rks_gpu",
    ),
    "get_veff_modified_uks_gpu": (
        "cc2cc.utils.modelscf_uks_gpu",
        "get_veff_modified_uks_gpu",
    ),
    "get_veff_grad_modified_uks_gpu": (
        "cc2cc.utils.modelscf_uks_gpu",
        "get_veff_grad_modified_uks_gpu",
    ),
}


def _load_symbol(name: str, table: dict[str, tuple[str, str]]) -> Any:
    module_name, attr_name = table[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


for _name in NO_LAZY_EXPORTS:
    _load_symbol(_name, NO_LAZY_EXPORTS)

__all__ = list(NO_LAZY_EXPORTS) + list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        return _load_symbol(name, _LAZY_EXPORTS)
    if name in _GPU_EXPORTS:
        if find_spec("cupy") is None:
            raise ImportError(
                f"GPU dependency cupy is not available. Cannot import {name}"
            )
        value = _load_symbol(name, _GPU_EXPORTS)
        if name not in __all__:
            __all__.append(name)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | set(NO_LAZY_EXPORTS) | set(_LAZY_EXPORTS) | set(_GPU_EXPORTS)
    )
