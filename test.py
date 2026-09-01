"""
Test the model.
Other parameter are from the argparse.
"""

import argparse

from cc2cc.utils import (
    DataRecord,
    ModelClass,
    add_args,
    gen_mole,
    print_computer_info,
)
from cc2cc.utils.parser import str2bool
from cc2cc.utils.run_helpers import build_record_path, release_memory, should_skip


def _run_one(mol, name, modeldict, data_record, args):
    if modeldict is None:
        raise ValueError("ModelClass is required for model validation mode.")

    from cc2cc.test_model_rks import test_model_rks
    from cc2cc.test_model_uks import test_model_uks

    runner = test_model_rks if mol.spin == 0 else test_model_uks
    runner(mol, name, modeldict, data_record, args)


def main():
    parser = argparse.ArgumentParser(
        description="Test the model. Other parameters are from the argparse."
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
        "--max_memory_gpu",
        type=int,
        default=4000,
        help="Maximum memory for GPU calculation in MB. Default is 4000.",
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
    parser.add_argument(
        "--s6",
        type=float,
        default=1.0,
        help="The s6 parameter for the D3 dispersion correction. Default is 1.0.",
    )
    parser.add_argument(
        "--s8",
        type=float,
        default=0.0,
        help="The s8 parameter for the D3 dispersion correction. Default is 1.0.",
    )
    parser.add_argument(
        "--a1",
        type=float,
        default=0.0,
        help="The a1 parameter for the D3 dispersion correction. Default is 0.0.",
    )
    parser.add_argument(
        "--a2",
        type=float,
        default=0.0,
        help="The a2 parameter for the D3 dispersion correction. Default is 0.0.",
    )
    parser.add_argument(
        "--s9",
        type=float,
        default=1.0,
        help="The s9 parameter for the D3 dispersion correction. Default is 1.0.",
    )
    parser.add_argument(
        "--alp",
        type=float,
        default=14.0,
        help="The alp parameter for the D3 dispersion correction. Default is 14.0.",
    )
    args = add_args(parser)

    print_computer_info(args.device)
    modeldict = ModelClass(args)
    modeldict.init_model(init_train=False)
    modeldict.eval()

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
            _run_one(
                mol,
                name,
                modeldict,
                data_record,
                args,
            )
        finally:
            del mol
            release_memory(args.device)

    if modeldict is not None:
        del modeldict
        release_memory(args.device)


if __name__ == "__main__":
    main()
