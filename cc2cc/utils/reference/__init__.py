"""Lazy exports for reference calculations."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "reference_b3lyp": ("reference_b3lyp", None),
    "reference_cc": ("reference_cc", None),
    "reference_hf": ("reference_hf", None),
    "reference_vv10": ("reference_vv10", None),
    "reference_zmp": ("reference_zmp", None),
    "test_data_dft": ("test_data_dft", None),
    "converged_b3lyp": ("reference_b3lyp", "converged_b3lyp"),
    "b3lyp_finite_difference_gradient": (
        "reference_b3lyp",
        "b3lyp_finite_difference_gradient",
    ),
    "finite_difference_gradient": ("reference_b3lyp", "finite_difference_gradient"),
    "get_dft_energy": ("reference_b3lyp", "get_dft_energy"),
    "get_dft_input": ("reference_b3lyp", "get_dft_input"),
    "get_dft_grad": ("reference_b3lyp", "get_dft_grad"),
    "CC_MAX_CYCLE": ("reference_cc", "CC_MAX_CYCLE"),
    "CC_CONV_TOL": ("reference_cc", "CC_CONV_TOL"),
    "CC_CONV_TOL_NORMT": ("reference_cc", "CC_CONV_TOL_NORMT"),
    "LAMBDA_MAX_CYCLE": ("reference_cc", "LAMBDA_MAX_CYCLE"),
    "LAMBDA_TOL": ("reference_cc", "LAMBDA_TOL"),
    "make_ccsd": ("reference_cc", "make_ccsd"),
    "ccsd_t_target_density": ("reference_cc", "ccsd_t_target_density"),
    "ccsd_t_target_gradient": ("reference_cc", "ccsd_t_target_gradient"),
    "get_cc_energy": ("reference_cc", "get_cc_energy"),
    "get_hf_energy": ("reference_hf", "get_hf_energy"),
    "eval_nlc_exc_density": ("reference_vv10", "eval_nlc_exc_density"),
    "post_dft_vv10_gradient": ("reference_vv10", "post_dft_vv10_gradient"),
    "get_vv10_data": ("reference_vv10", "get_vv10_data"),
    "get_post_vv10_data": ("reference_vv10", "get_post_vv10_data"),
    "get_scf_vv10_data": ("reference_vv10", "get_scf_vv10_data"),
    "post_vv10_total_energy": ("reference_vv10", "post_vv10_total_energy"),
    "post_vv10_finite_difference_gradient": (
        "reference_vv10",
        "post_vv10_finite_difference_gradient",
    ),
    "LEVEL_SHIFT": ("reference_zmp", "LEVEL_SHIFT"),
    "MAX_CYCLE": ("reference_zmp", "MAX_CYCLE"),
    "RHO_REL_TOL": ("reference_zmp", "RHO_REL_TOL"),
    "I_VALUE_REL_TOL": ("reference_zmp", "I_VALUE_REL_TOL"),
    "I_VALUE_ABS_TOL": ("reference_zmp", "I_VALUE_ABS_TOL"),
    "DIIS_SPACE": ("reference_zmp", "DIIS_SPACE"),
    "get_zmp": ("reference_zmp", "get_zmp"),
    "zmp_baseline_gradient": ("reference_zmp", "zmp_baseline_gradient"),
    "TestDataDFT": ("test_data_dft", "TestDataDFT"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
