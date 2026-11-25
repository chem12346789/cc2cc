"""
Benchmark DFT calculations.
Other parameter are from the argparse.
"""

import argparse

from cc2cc.utils.helper.runtime.computer_info import print_computer_info
from cc2cc.utils.helper.workflow.DataRecord import DataRecord
from cc2cc.utils.mol import gen_mole
from cc2cc.utils.parser import add_args, str2bool
from cc2cc.utils.reference.test_data_dft import TestDataDFT
from cc2cc.utils.helper.workflow.run_helpers import (
    build_record_path,
    release_memory,
    should_skip,
)


def _benchmark_dft(
    mol,
    name,
    data_record,
    benchmark_method,
    benchmark_disp,
    use_gpu: bool,
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
            use_gpu=use_gpu,
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
        "--max_cycle",
        type=int,
        default=250,
        help="Maximum number of SCF cycles. Default is 250 and -1 for no iteration.",
    )
    parser.add_argument(
        "--if_grad",
        type=str2bool,
        default=False,
        help="Whether to calculate the force. Default is False.",
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
        "--conv_tol",
        type=float,
        default=1e-7,
        help="SCF energy convergence tolerance in Hartree. Default is 1e-7.",
    )
    parser.add_argument(
        "--conv_tol_grad",
        type=float,
        default=1e-3,
        help="SCF orbital-gradient convergence tolerance. Default is 1e-3.",
    )
    parser.add_argument(
        "--cpu",
        type=str2bool,
        default=False,
        help="Force the use of CPU even if GPU is available. Default is False.",
    )
    parser.add_argument(
        "--if_rotate",
        type=str2bool,
        default=False,
        help="Whether to use rotation. Default is False.",
    )
    parser.add_argument(
        "--if_rotate_random",
        type=str2bool,
        default=False,
        help="Whether to use random rotation. Default is False.",
    )
    args = add_args(parser)

    print_computer_info(args.device)

    if args.benchmark_disp is None:
        args.benchmark_disp = [None] * len(args.benchmark_method)

    if len(args.benchmark_disp) == 1 and len(args.benchmark_method) > 1:
        args.benchmark_disp = args.benchmark_disp * len(args.benchmark_method)

    if len(args.benchmark_method) != len(args.benchmark_disp):
        raise ValueError(
            "`--benchmark_method` and `--benchmark_disp` must have same length."
        )

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
                use_gpu=args.device == "cuda",
            )
        finally:
            del mol
            release_memory(args.device)


if __name__ == "__main__":
    main()
