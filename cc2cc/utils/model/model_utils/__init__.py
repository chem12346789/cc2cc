"""Public utility API for cc2cc.utils.model.model_utils."""

from __future__ import annotations

from typing import Any

from cc2cc.utils._lazy_exports import load_lazy_export as _load_lazy_export

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "Transformer": ("cc2cc.utils.model.model_utils.model_transformer", "Transformer"),
    "DenseNet": ("cc2cc.utils.model.model_utils.model_dense", "DenseNet"),
    "E3nn": ("cc2cc.utils.model.model_utils.model_e3nn", "E3nn"),
    "E3nnC": ("cc2cc.utils.model.model_utils.model_e3nn_correct", "E3nn"),
}
NO_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {}
_GPU_EXPORTS: dict[str, tuple[str, str | None]] = {}


for _name in NO_LAZY_EXPORTS:
    _load_lazy_export(_name, globals(), NO_LAZY_EXPORTS)

__all__ = list(NO_LAZY_EXPORTS) + list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    return _load_lazy_export(name, globals(), _LAZY_EXPORTS, _GPU_EXPORTS)


def __dir__() -> list[str]:
    return sorted(
        set(globals()) | set(NO_LAZY_EXPORTS) | set(_LAZY_EXPORTS) | set(_GPU_EXPORTS)
    )
