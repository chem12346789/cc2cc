"""Public utility API for cc2cc.

Most CPU exports are loaded eagerly, while SCF helper functions and GPU exports
remain lazy.
"""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

NO_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "add_args": ("cc2cc.utils.parser", "add_args"),
    "config_list": ("cc2cc.utils.parser", "config_list"),
    "process_config": ("cc2cc.utils.parser", "process_config"),
    "gen_mole": ("cc2cc.utils.mol", "gen_mole"),
    "AU2KCALMOL": ("cc2cc.utils.helper.units", "AU2KCALMOL"),
    "AU2DEBYE": ("cc2cc.utils.helper.units", "AU2DEBYE"),
    "DATA_PATH": ("cc2cc.utils.helper.runtime.env_var", "DATA_PATH"),
    "MAIN_PATH": ("cc2cc.utils.helper.runtime.env_var", "MAIN_PATH"),
    "print_computer_info": (
        "cc2cc.utils.helper.runtime.computer_info",
        "print_computer_info",
    ),
    "TestDataDFT": ("cc2cc.utils.reference.test_data_dft", "TestDataDFT"),
    "diff_rho": ("cc2cc.utils.helper.chemistry.density_metrics", "diff_rho"),
    "eval_nlc_exc_density": (
        "cc2cc.utils.reference.reference_vv10",
        "eval_nlc_exc_density",
    ),
    "Timer": ("cc2cc.utils.helper.runtime.timer", "Timer"),
    "DataRecord": ("cc2cc.utils.helper.workflow.DataRecord", "DataRecord"),
    "ModelClass": ("cc2cc.utils.ModelClass", "ModelClass"),
    "Grid": ("cc2cc.utils.Grids", "Grid"),
    "GridCPU": ("cc2cc.utils.Grids", "GridCPU"),
    "build_record_path": (
        "cc2cc.utils.helper.workflow.run_helpers",
        "build_record_path",
    ),
    "release_memory": ("cc2cc.utils.helper.workflow.run_helpers", "release_memory"),
    "should_skip": ("cc2cc.utils.helper.workflow.run_helpers", "should_skip"),
}

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "get_veff_modified": (
        "cc2cc.utils.modelscf",
        "get_veff_modified",
    ),
    "get_veff_grad_modified": (
        "cc2cc.utils.modelscf",
        "get_veff_grad_modified",
    ),
}

_GPU_EXPORTS: dict[str, tuple[str, str | None]] = {
    "GridGPU": ("cc2cc.utils.GridsGPU", "GridGPU"),
    "get_veff_modified_gpu": (
        "cc2cc.utils.modelscf_gpu",
        "get_veff_modified_gpu",
    ),
    "get_veff_grad_modified_gpu": (
        "cc2cc.utils.modelscf_gpu",
        "get_veff_grad_modified_gpu",
    ),
}


for _name in NO_LAZY_EXPORTS:
    _load_lazy_export(_name, globals(), NO_LAZY_EXPORTS)

__all__ = list(NO_LAZY_EXPORTS) + list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS, _GPU_EXPORTS)


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | set(NO_LAZY_EXPORTS) | set(_LAZY_EXPORTS) | set(_GPU_EXPORTS)
    )
