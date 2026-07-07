"""Public utility API for cc2cc.utils.model.model_utils."""

from __future__ import annotations
from importlib import import_module
from importlib.util import find_spec
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Transformer": ("cc2cc.utils.model.model_utils.model_transformer", "Transformer"),
    "DenseNet": ("cc2cc.utils.model.model_utils.model_dense", "DenseNet"),
    "E3nn": ("cc2cc.utils.model.model_utils.model_e3nn", "E3nn"),
}
NO_LAZY_EXPORTS: dict[str, tuple[str, str]] = {}
_GPU_EXPORTS: dict[str, tuple[str, str]] = {}


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
