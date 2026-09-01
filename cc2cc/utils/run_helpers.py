"""Shared helpers for lightweight test/benchmark runners."""

from __future__ import annotations

import gc
from pathlib import Path

from cc2cc.utils.env_var import MAIN_PATH

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def release_memory(device) -> None:
    gc.collect()
    if str(device).lower() not in ("cpu", "none"):
        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        if torch is not None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def build_record_path(args) -> Path:
    if "gmtkn-diet" in args.dataset.lower():
        base = f"ccdft_{args.basis}_{args.load}_gmtkn-def2"
    else:
        base = f"ccdft_{args.basis}_{args.load}_{args.dataset}"
    if len(args.name_mol_input) == 1:
        base = f"{base}_{args.name_mol_input[0]}"
    dir_path = MAIN_PATH / "validate" / f"{args.basis}_{args.load}" / f"{args.load_epoch}"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"{base}.csv"


def should_skip(name: str, data_record, args) -> bool:
    return args.if_continue and name in data_record.df_dict.get("name", [])
