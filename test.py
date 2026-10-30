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
        "--grid_response",
        type=int,
        choices=(0, 1),
        default=0,
        help="Include grid response in force calculations: 1 enables, 0 disables (default).",
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
        "--require_convergence",
        type=str2bool,
        default=False,
        help="Reject unconverged SCF results. Default is False.",
    )
    parser.add_argument(
        "--test_mode",
        choices=(
            "scf",
            "post-b3lyp",
            "post-b3lyp-d3",
            "post-b3lyp-nl",
            "post-b3lyp-vv10",
        ),
        default="scf",
        help="Run model SCF (default), or evaluate once without model SCF on a "
        "cached density: B3LYP for post-b3lyp, post-b3lyp-d3, and "
        "post-b3lyp-nl; self-consistent B3LYP-VV10 for "
        "post-b3lyp-vv10. The D3 mode reports D3BJ energy, and the NL mode "
        "reports the post-SCF VV10 energy. Missing references are computed by "
        "reference SCF and cached. Post-SCF modes are energy-only.",
    )
    parser.add_argument(
        "--baseline",
        choices=("b3lyp-vv10",),
        default=None,
        help="Include self-consistent VV10 in the model's B3LYP baseline. "
        "Use --if_continue 0 to recompute previously saved results.",
    )
    parser.add_argument(
        "--max_memory_gpu",
        type=int,
        default=4000,
        help="Maximum memory for GPU calculation in MB. Default is 4000.",
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
    parser.add_argument(
        "--s6",
        type=float,
        default=None,
        help="The s6 parameter for the D3 dispersion correction. Default is None, which means it is the DEFAULT value.",
    )
    parser.add_argument(
        "--s8",
        type=float,
        default=None,
        help="The s8 parameter for the D3 dispersion correction. Default is None, which means it is the DEFAULT value.",
    )
    parser.add_argument(
        "--a1",
        type=float,
        default=None,
        help="The a1 parameter for the D3 dispersion correction. Default is None, which means it is the DEFAULT value.",
    )
    parser.add_argument(
        "--a2",
        type=float,
        default=None,
        help="The a2 parameter for the D3 dispersion correction. Default is None, which means it is the DEFAULT value.",
    )
    parser.add_argument(
        "--s9",
        type=float,
        default=None,
        help="The s9 parameter for the D3 dispersion correction. Default is None, which means it is the DEFAULT value.",
    )
    parser.add_argument(
        "--alp",
        type=float,
        default=None,
        help="The alp parameter for the D3 dispersion correction. Default is None, which means it is the DEFAULT value.",
    )
    args = add_args(parser)
    if args.test_mode != "scf":
        if args.if_grad:
            parser.error("Post-SCF density modes do not support nuclear gradients.")
        if args.if_rotate or args.if_rotate_random:
            parser.error("Post-SCF density modes require the cached molecular orientation.")
        if args.test_mode in (
            "post-b3lyp",
            "post-b3lyp-d3",
            "post-b3lyp-nl",
        ) and args.baseline is not None:
            parser.error(
                "The B3LYP-density post-SCF modes cannot use --baseline b3lyp-vv10."
            )

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

            print(f"\n\n\n{'-' * 80}\nRUN: {name}\n{'-' * 80}", flush=True)
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
