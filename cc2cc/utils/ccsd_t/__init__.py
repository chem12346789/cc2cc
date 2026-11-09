"""Lazy exports for PySCF CCSD(T) density and lambda helpers."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "pyscf_ccsd_t_rdm": ("pyscf_ccsd_t_rdm", None),
    "pyscf_uccsd_t_lambda": ("pyscf_uccsd_t_lambda", None),
    "pyscf_uccsd_t_u_gamma1_intermediates": (
        "pyscf_uccsd_t_u_gamma1_intermediates",
        None,
    ),
    "pyscf_uccsd_t_u_gamma2_intermediates": (
        "pyscf_uccsd_t_u_gamma2_intermediates",
        None,
    ),
    "gamma1_intermediates": ("pyscf_ccsd_t_rdm", "gamma1_intermediates"),
    "t3_symm_ip_py": ("pyscf_ccsd_t_rdm", "t3_symm_ip_py"),
    "triples_block_size": ("pyscf_uccsd_t_lambda", "triples_block_size"),
    "same_spin_v": ("pyscf_uccsd_t_lambda", "same_spin_v"),
    "same_spin_r6_divide": ("pyscf_uccsd_t_lambda", "same_spin_r6_divide"),
    "mixed_spin_v": ("pyscf_uccsd_t_lambda", "mixed_spin_v"),
    "mixed_spin_divide": ("pyscf_uccsd_t_lambda", "mixed_spin_divide"),
    "run_fused": ("pyscf_uccsd_t_lambda", "run_fused"),
    "kernel": ("pyscf_uccsd_t_lambda", "kernel"),
    "make_intermediates": ("pyscf_uccsd_t_lambda", "make_intermediates"),
    "update_lambda": ("pyscf_uccsd_t_lambda", "update_lambda"),
    "u_gamma1_intermediates": (
        "pyscf_uccsd_t_u_gamma1_intermediates",
        "u_gamma1_intermediates",
    ),
    "u_gamma2_intermediates": (
        "pyscf_uccsd_t_u_gamma2_intermediates",
        "u_gamma2_intermediates",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
