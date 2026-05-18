"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
import gc
from pathlib import Path

import torch

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils import ModelClass, DataRecord
from cc2cc.utils import MAIN_PATH
from cc2cc.test_model_rks import test_model_rks
from cc2cc.test_model_uks import test_model_uks


def _build_record_path(args) -> Path:
    base = f"ccdft_{args.basis}_{args.load}_{args.dataset}"
    if len(args.name_mol_input) == 1:
        base = f"{base}_{args.name_mol_input[0]}"
    return MAIN_PATH / f"validate/{base}.csv"


def _should_skip(name: str, data_record, args) -> bool:
    if not args.if_continue:
        return False
    return ("name" in data_record.df_dict) and (name in data_record.df_dict["name"])


def _run_one(mol, name, modeldict, data_record, args):
    if mol.spin == 0:
        test_model_rks(mol, name, modeldict, data_record, args)
    else:
        test_model_uks(mol, name, modeldict, data_record, args)


def _release_memory(device) -> None:
    gc.collect()
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    print_computer_info(args.device)

    modeldict = ModelClass(args)
    if "test" not in args.load:
        modeldict.init_model(if_validate=True)
        modeldict.eval()

    data_record = DataRecord(
        _build_record_path(args),
        if_continue=args.if_continue,
    )
    error_molecule = []

    for name_mol in args.name_mol:
        name = f"{name_mol}_{args.basis}"
        mol = None
        try:
            mol = gen_mole(
                name_mol,
                args.basis,
                dataset_name=args.dataset,
                if_rotate=args.if_rotate,
                if_rotate_random=args.if_rotate_random,
            )

            if mol is None:
                print(f"SKIP: {name}")
                continue

            if _should_skip(name, data_record, args):
                print(f"SKIP: {name}")
                continue

            _run_one(mol, name, modeldict, data_record, args)
        finally:
            del mol
            _release_memory(args.device)

    del modeldict
    _release_memory(args.device)
    print(f"Error molecule: {error_molecule}", flush=True)


if __name__ == "__main__":
    main()
