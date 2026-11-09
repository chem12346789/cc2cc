"""Shared loader for lazy utility-package exports."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any


def load_lazy_export(
    name: str,
    namespace: dict[str, Any],
    exports: dict[str, tuple[str, str | None]],
    gpu_exports: dict[str, tuple[str, str | None]] | None = None,
) -> Any:
    if name in exports:
        module_name, attribute = exports[name]
        is_gpu = False
    elif gpu_exports is not None and name in gpu_exports:
        if find_spec("cupy") is None:
            raise ImportError(
                f"GPU dependency cupy is not available. Cannot import {name}"
            )
        module_name, attribute = gpu_exports[name]
        is_gpu = True
    else:
        raise AttributeError(
            f"module {namespace['__name__']!r} has no attribute {name!r}"
        )

    module = (
        import_module(module_name, namespace["__name__"])
        if module_name.startswith(".")
        else import_module(module_name)
    )
    value = module if attribute is None else getattr(module, attribute)
    namespace[name] = value
    if is_gpu and name not in namespace["__all__"]:
        namespace["__all__"].append(name)
    return value
