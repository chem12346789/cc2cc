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
    model_load = args.load
    test_mode = getattr(args, "test_mode", "scf")
    if test_mode != "scf":
        model_load = f"{model_load}-{test_mode}"
    if "gmtkn-diet" in args.dataset.lower():
        base = f"ccdft_{args.basis}_{model_load}_gmtkn-def2"
    else:
        base = f"ccdft_{args.basis}_{model_load}_{args.dataset}"
    if len(args.name_mol_input) == 1:
        base = f"{base}_{args.name_mol_input[0]}"
    dir_path = MAIN_PATH / "validate" / f"{args.basis}_{model_load}" / f"{args.load_epoch}"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"{base}.csv"


def load_post_scf_density(mol, name, test_mode, *, grid_level=None):
    """Load a reference density, caching missing self-consistent VV10 references."""
    import numpy as np

    from cc2cc.utils.env_var import DATA_PATH, DATA_TEST_PATH

    key = "dm1_scf_vv10" if test_mode == "post-scf-b3lyp-vv10" else "dm1_dft"
    paths = (
        DATA_PATH / f"data_{name}_addon.npz",
        DATA_PATH / f"data_{name}.npz",
        DATA_TEST_PATH / f"{name}_cc.npz",
    )
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            cache_key = key
            if key == "dm1_dft" and path == paths[-1] and "dm1_dft-b3lyp" in data:
                cache_key = "dm1_dft-b3lyp"
            if cache_key not in data:
                continue
            if "mol_corr" in data and not np.allclose(
                mol.atom_coords(), data["mol_corr"], rtol=0, atol=1e-6
            ):
                raise ValueError(f"Cached density geometry differs from {name}: {path}")
            if key == "dm1_scf_vv10" and "e_scf_vv10" not in data:
                continue
            return data[cache_key]
    if key == "dm1_scf_vv10":
        from tempfile import TemporaryDirectory

        from pyscf.dft.gen_grid import Grids

        from cc2cc.utils.b3lyp_reference import converged_b3lyp

        print(f"Generating missing B3LYP-VV10 reference for {name}.")
        grids = Grids(mol)
        if grid_level is not None:
            grids.level = grid_level
        reference = converged_b3lyp(mol, nlc="vv10", grids=grids, verbose=4)
        if not reference.converged:
            reference.level_shift = 1
            reference.kernel(dm0=reference.make_rdm1())
        if not reference.converged:
            raise RuntimeError(f"B3LYP-VV10 reference SCF did not converge for {name}.")
        density = reference.make_rdm1()
        addon_path = paths[0]
        if addon_path.exists():
            with np.load(addon_path, allow_pickle=True) as data:
                cached = dict(data.items())
            if "mol_corr" in cached and not np.allclose(
                mol.atom_coords(), cached["mol_corr"], rtol=0, atol=1e-6
            ):
                raise ValueError(f"Cached density geometry differs from {name}: {addon_path}")
        else:
            cached = {}
        cached.update(
            dm1_scf_vv10=density,
            e_scf_vv10=float(reference.e_tot),
            mol_corr=mol.atom_coords(),
        )
        addon_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=addon_path.parent) as directory:
            temporary = Path(directory) / addon_path.name
            np.savez(temporary, **cached)
            temporary.replace(addon_path)
        print(f"Cached B3LYP-VV10 density and energy in {addon_path}.")
        return density
    raise FileNotFoundError(
        f"{test_mode} requires cached {key} for {name}; searched "
        + ", ".join(str(path) for path in paths)
        + ". No SCF was run."
    )


def should_skip(name: str, data_record, args) -> bool:
    return args.if_continue and name in data_record.df_dict.get("name", [])
