"""
Benchmark DFT calculations.
Other parameter are from the argparse.
"""

import argparse

from cc2cc.utils import (
    DataRecord,
    TestDataDFT,
    add_args,
    gen_mole,
    print_computer_info,
)
from cc2cc.utils.run_helpers import build_record_path, release_memory, should_skip


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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DFT calculations. Other parameters are from the argparse."
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
    args = add_args(parser)

    print_computer_info(args.device)

    if args.benchmark_disp is None:
        args.benchmark_disp = [None] * len(args.benchmark_method)

    if len(args.benchmark_disp) == 1 and len(args.benchmark_method) > 1:
        args.benchmark_disp = args.benchmark_disp * len(args.benchmark_method)

    if len(args.benchmark_method) != len(args.benchmark_disp):
        raise ValueError("`--benchmark_method` and `--benchmark_disp` must have same length.")

    data_record = DataRecord(
        build_record_path(args),
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

            if should_skip(name, data_record, args):
                print(f"SKIP: {name}")
                continue

            print(f"RUN: {name}", flush=True)
            _benchmark_dft(
                mol,
                name,
                data_record,
                args.benchmark_method,
                args.benchmark_disp,
            )
        finally:
            del mol
            release_memory(args.device)


if __name__ == "__main__":
    main()
