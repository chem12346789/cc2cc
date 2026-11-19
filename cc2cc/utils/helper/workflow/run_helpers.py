"""Shared helpers for lightweight test/benchmark runners."""

from __future__ import annotations

import gc
from pathlib import Path

from cc2cc.utils.helper.runtime.env_var import MAIN_PATH

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


def load_post_scf_density(
    mol, name, test_mode, *, unrestricted, grid_level=None, use_gpu=None,
):
    """Load or generate reference densities exclusively in the test cache."""
    from tempfile import TemporaryDirectory

    import numpy as np
    from pyscf.dft.gen_grid import Grids

    from cc2cc.utils.reference.reference_b3lyp import converged_dft
    from cc2cc.utils.helper.runtime.env_var import DATA_TEST_PATH

    xc = test_mode.split("-")[1]
    vv10 = test_mode.endswith("-vv10")
    functional = f"{xc}-vv10" if vv10 else xc
    key = f"dm1_dft-{functional}"
    energy_key = f"e_dft-{functional}"
    if xc == "b3lyp" and vv10:
        key, energy_key = "dm1_scf_vv10", "e_scf_vv10"
    path = DATA_TEST_PATH / f"{name}_dft.npz"
    cached = {}
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            cached = dict(data.items())
        if "mol_corr" in cached and not np.allclose(
            mol.atom_coords(), cached["mol_corr"], rtol=0, atol=1e-6
        ):
            raise ValueError(f"Cached density geometry differs from {name}: {path}")
        cache_key = key
        if xc == "b3lyp" and not vv10 and key not in cached:
            cache_key = "dm1_dft"
        if cache_key in cached and energy_key in cached:
            if cached[cache_key].ndim == (3 if unrestricted else 2):
                return cached[cache_key]

    method = functional.upper()
    if use_gpu is None:
        use_gpu = torch is not None and torch.cuda.is_available()
    print(f"Generating missing {method} reference for {name} on {'GPU' if use_gpu else 'CPU'}.")
    grids = Grids(mol)
    if grid_level is not None:
        grids.level = grid_level
    dm0 = cached.get(f"dm1_dft-{xc}")
    if dm0 is None and xc == "b3lyp":
        dm0 = cached.get("dm1_dft")
    if dm0 is not None and dm0.ndim != (3 if unrestricted else 2):
        dm0 = None
    reference = converged_dft(
        mol, dm0=dm0, xc=xc, unrestricted=unrestricted,
        nlc="vv10" if vv10 else "", grids=grids, verbose=4, use_gpu=use_gpu,
    )
    if not reference.converged:
        reference.level_shift = 1
        reference.kernel(dm0=reference.make_rdm1())
    if not reference.converged:
        raise RuntimeError(f"{method} reference SCF did not converge for {name}.")
    density = reference.make_rdm1()
    if use_gpu:
        density = cp.asnumpy(density)
    cached.update({
        key: density,
        energy_key: float(reference.e_tot),
        "mol_corr": mol.atom_coords(),
    })
    if xc == "b3lyp" and not vv10:
        cached["dm1_dft"] = density
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        np.savez(temporary, **cached)
        temporary.replace(path)
    print(f"Cached {method} density and energy in {path}.")
    return density


def load_post_scf_vv10(mf, name, density, *, unrestricted):
    """Cache the post-SCF VV10 energy for an exact density and NLC grid."""
    from hashlib import sha256
    from tempfile import TemporaryDirectory

    import numpy as np

    from cc2cc.utils.helper.runtime.env_var import DATA_TEST_PATH

    def as_numpy(value):
        if cp is not None and isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
        return np.asarray(value)

    fingerprint = sha256(type(mf._numint).__module__.encode())
    for value in (
        mf.mol._atm, mf.mol._bas, mf.mol._env,
        density, mf.nlcgrids.coords, mf.nlcgrids.weights,
    ):
        array = as_numpy(value)
        fingerprint.update(str((array.shape, array.dtype)).encode())
        fingerprint.update(array.tobytes())
    fingerprint = fingerprint.hexdigest()
    path = DATA_TEST_PATH / f"{name}_dft.npz"
    cached = {}
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            cached = dict(data.items())
        if (
            "enlc_post_vv10" in cached
            and cached.get("enlc_post_vv10_fingerprint") == fingerprint
        ):
            print(f"Using cached post-SCF VV10 correction for {name}.")
            return float(cached["enlc_post_vv10"])

    dm_tot = density[0] + density[1] if unrestricted else density
    _, enlc, _ = mf._numint.nr_nlc_vxc(
        mf.mol, mf.nlcgrids, "vv10", dm_tot, max_memory=mf.max_memory,
    )
    enlc = float(enlc)
    cached.update({
        "enlc_post_vv10": enlc,
        "enlc_post_vv10_fingerprint": fingerprint,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        np.savez(temporary, **cached)
        temporary.replace(path)
    print(f"Cached post-SCF VV10 correction for {name}: {enlc:.12g} Ha.")
    return enlc


def should_skip(name: str, data_record, args) -> bool:
    return args.if_continue and name in data_record.df_dict.get("name", [])
