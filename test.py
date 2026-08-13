"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from pathlib import Path
import gc

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from cc2cc.utils import (
    MAIN_PATH,
    DataRecord,
    ModelClass,
    TestDataDFT,
    add_args,
    gen_mole,
    print_computer_info,
)
from cc2cc.utils.parser import str2bool


def _release_memory(device) -> None:
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


def _build_record_path(args) -> Path:
    if "gmtkn-diet" in args.dataset.lower():
        base = f"ccdft_{args.basis}_{args.load}_gmtkn-def2"
    else:
        base = f"ccdft_{args.basis}_{args.load}_{args.dataset}"
    if len(args.name_mol_input) == 1:
        base = f"{base}_{args.name_mol_input[0]}"
    dir_path = (
        MAIN_PATH / "validate" / f"{args.basis}_{args.load}" / f"{args.load_epoch}"
    )
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"{base}.csv"


def _should_skip(name: str, data_record, args) -> bool:
    return args.if_continue and name in data_record.df_dict.get("name", [])


def _benchmark_dft(
    mol,
    name,
    data_record,
    benchmark_method,
    benchmark_disp,
) -> None:
    record = {"name": name}
    for xc_code, disp in zip(benchmark_method, benchmark_disp):
        if disp in ("", "None", "none", "null", "Null"):
            disp = None
        test_data = TestDataDFT(
            mol,
            name,
            xc_code=xc_code,
            disp=disp,
        )
        xc_code_disp = xc_code if disp is None else f"{xc_code}-{disp}"
        record.update(
            {
                f"{xc_code_disp}_ene": test_data.e_dft,
                f"{xc_code_disp}_dipole_x": test_data.dft_dipole[0],
                f"{xc_code_disp}_dipole_y": test_data.dft_dipole[1],
                f"{xc_code_disp}_dipole_z": test_data.dft_dipole[2],
            }
        )
    data_record.add_data(record)
    data_record.save_csv()


def _is_benchmark_mode(args) -> bool:
    return any(flag in args.load.lower() for flag in ("test", "benchmark"))


def _run_one(mol, name, modeldict, data_record, args, benchmark_mode: bool):
    if benchmark_mode:
        _benchmark_dft(
            mol,
            name,
            data_record,
            args.benchmark_method,
            args.benchmark_disp,
        )
        return

    if modeldict is None:
        raise ValueError("ModelClass is required for model validation mode.")

    from cc2cc.test_model_rks import test_model_rks
    from cc2cc.test_model_uks import test_model_uks

    runner = test_model_rks if mol.spin == 0 else test_model_uks
    runner(mol, name, modeldict, data_record, args)


def main():
    parser = argparse.ArgumentParser(
        description="Test the model or benchmark DFT calculations. Other parameters are from the argparse."
    )
    parser.add_argument(
        "--benchmark_method",
        type=str,
        nargs="+",
        default=["b3lyp"],
        help="Benchmark method for DFT calculations. Default is b3lyp.",
    )
    parser.add_argument(
        "--benchmark_disp",
        type=str,
        nargs="+",
        default=None,
        help="Dispersion correction for benchmark DFT calculations. Default is None.",
    )
    parser.add_argument(
        "--if_grad",
        type=str2bool,
        default=False,
        help="Whether to calculate the force. Default is False.",
    )
    args = add_args(parser)

    print_computer_info(args.device)

    benchmark_mode = _is_benchmark_mode(args)
    modeldict = None
    if not benchmark_mode:
        modeldict = ModelClass(args)
        modeldict.init_model(if_validate=True)
        modeldict.eval()
    else:
        print("Benchmark mode: skip model loading.", flush=True)

    data_record = DataRecord(
        _build_record_path(args),
        if_continue=args.if_continue,
    )

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

            print(f"RUN: {name}", flush=True)
            _run_one(
                mol,
                name,
                modeldict,
                data_record,
                args,
                benchmark_mode=benchmark_mode,
            )
        finally:
            del mol
            _release_memory(args.device)

    if modeldict is not None:
        del modeldict
        _release_memory(args.device)


if __name__ == "__main__":
    main()
