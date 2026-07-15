"""
@package docstring
Documentation for this module.

More details.
"""

import argparse


def str2bool(v):
    """
    Function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def gen_name_args(
    name_args, args_dataset, args_name_mol_reverse=False, if_exclude=False
):
    """
    Function to generate name args
    """
    from cc2cc.utils.mol import dataset

    dataset_dict = dataset[args_dataset]

    if name_args is None:
        name_mol_new = dataset_dict["molecule"]
    elif len(name_args) == 0:
        name_mol_new = []
    else:
        name_mol_new = []
        for i in range(len(name_args)):
            if name_args[i].startswith("molecule"):
                if name_args[i] in dataset_dict.keys():
                    for extend_mol in dataset_dict[name_args[i]]:
                        if isinstance(dataset_dict[extend_mol], str):
                            name_mol_new.append(dataset_dict[extend_mol])
                        else:
                            name_mol_new.append(extend_mol)
                elif if_exclude:
                    print(f"Warning: {name_args[i]} is not in the dataset. ")
                else:
                    raise ValueError(
                        f"Invalid molecule name: {name_args[i]}. "
                        f"Please use a valid molecule name in {dataset_dict.keys()}"
                    )
            else:
                if isinstance(dataset_dict[name_args[i]], str):
                    name_mol_new.append(dataset_dict[name_args[i]])
                else:
                    name_mol_new.append(name_args[i])

        # remove duplicates
        name_mol = []
        for i in name_mol_new:
            if i not in name_mol:
                name_mol.append(i)
        name_mol_new = name_mol

    # sort the name_mol_new by the length of the molecule then by the name
    # this is to ensure that the training process will be reproducible
    name_mol_new.sort(key=lambda x: (len(dataset_dict[x]), x))
    if args_name_mol_reverse:
        name_mol_new = name_mol_new[::-1]

    return name_mol_new


def add_args(parser: argparse.ArgumentParser):
    """
    Documentation for a function.

    More details.
    """
    # ========= Arguments ==========
    # for data generation
    parser.add_argument(
        "--name_mol",
        "-m",
        nargs="+",
        type=str,
        default=[],
        help="Name of molecule. Default is None (all the dataset).",
    )

    parser.add_argument(
        "--name_mol_reverse",
        type=str2bool,
        default=False,
        help="Whether to reverse the order of the molecule names. Default is False.",
    )

    parser.add_argument(
        "--md_number",
        type=int,
        default=0,
        help="MD frame number to generate the data. Default is 0.",
    )

    parser.add_argument(
        "--mp_number",
        type=int,
        default=0,
        help="Number of training cycles. Default is 1.",
    )

    parser.add_argument(
        "--mp_total",
        type=int,
        default=3,
        help="Total number of training cycles. Default is 3.",
    )

    parser.add_argument(
        "--if_eval",
        type=str2bool,
        default=False,
        help="Whether to use the evaluation mode in generating the data. Default is False.",
    )

    parser.add_argument(
        "--grid_level",
        type=int,
        default=4,
        help="Grid level for the calculation.",
    )

    parser.add_argument(
        "--basis",
        type=str,
        default="cc-pVDZ",
        help="Basis set for the calculation.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mol",
        help="Name of the dataset. Default is mol (training and testing).",
    )

    parser.add_argument(
        "--check_convergence",
        type=str2bool,
        default=True,
        help="Whether to check the convergence of the wave function. "
        "Default is True.",
    )

    # ========== Arguments ==========
    # for training
    parser.add_argument(
        "--rho_input",
        type=str,
        default="dft",
        choices=["dft", "dft_d3bj", "zmp"],
        help="Type of input density. ",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["MSELoss", "L1Loss"],
        default="MSELoss",
        help="Loss function for the energy. "
        "Default is MSELoss. Other options are L1Loss.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="transformer+dense_mix_e3nn_4",
        help="Model for training.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for the training. Default is cuda.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=10000,
        help="Number of epoch for training. Default is 10000.",
    )

    parser.add_argument(
        "--distributed",
        default=False,
        type=str2bool,
        help="Whether to use distributed training. Default is False.",
    )

    parser.add_argument(
        "--activation_memory_budget",
        type=float,
        default=None,
        help="Deprecated placeholder for backward compatibility (currently unused).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Precision for the training. Default is float64.",
    )

    parser.add_argument(
        "--if_compile",
        type=str2bool,
        default=False,
        help="Enable torch.compile for the model. Default is False.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the training. Default is 1e-4.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["AdamW", "Adafactor"],
        help="Optimizer for the training. Default is AdamW.",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="constant",
        choices=["cosine", "constant", "cosine_warm"],
        help="Learning rate scheduler. Default is constant.",
    )

    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=1e-8,
        help="Minimum learning rate for cosine scheduler. Default is 1e-8.",
    )

    parser.add_argument(
        "--cosine_T",
        type=int,
        default=16,
        help="Number of periods for the cosine scheduler. Default is 16.",
    )

    parser.add_argument(
        "--cosine_T_mult",
        type=int,
        default=1,
        help="Multiplicative factor for the period of the cosine scheduler. Default is 1.",
    )

    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="Max norm for the gradient. Default is 1.0.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the training. Default is None (no seed). ",
    )

    parser.add_argument(
        "--deterministic",
        type=str2bool,
        default=False,
        help="Enable deterministic CUDA backend settings (slower but reproducible).",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for the optimizer. Default is 1e-3.",
    )

    parser.add_argument(
        "--eval_step",
        type=int,
        default=1,
        help="Step for evaluation. Default is 1.",
    )

    parser.add_argument(
        "--if_relative_weight",
        type=str2bool,
        default=False,
        help="Whether to use relative weight for the loss function. Default is False.",
    )

    parser.add_argument(
        "--if_relative_weight_abs",
        type=str2bool,
        default=False,
        help="Whether to use relative weight for the absolute loss function. Default is False.",
    )

    parser.add_argument(
        "--atomic_weighting",
        type=int,
        default=1,
        help="Weighting scheme for atomic energy. Default is 1 (with 20 copies).",
    )

    parser.add_argument(
        "--output_target",
        type=str,
        default="tol_delta_grids",
        choices=[
            "tol_delta_grids",
            "tol_delta_grids_l",
            "tol_delta_grids_l_erf",
            "exc_cc_grids",
            "b3lyp",
        ],
        help="Target for the output. Default is tol_delta_grids.",
    )

    parser.add_argument(
        "--loss_multiplier",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )

    parser.add_argument(
        "--loss_multiplier_abs",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )

    parser.add_argument(
        "--loss_multiplier_atomic",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )

    parser.add_argument(
        "--loss_multiplier_grad",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )

    parser.add_argument(
        "--if_grad",
        type=str2bool,
        default=False,
        help="Whether to calculate the gradient. Default is False.",
    )

    parser.add_argument(
        "--if_atomic",
        type=str2bool,
        default=False,
        help="Whether to calculate the atomic energy. Default is False.",
    )

    parser.add_argument(
        "--if_abs",
        type=str2bool,
        default=False,
        help="Whether to calculate the absolute energy. Default is False.",
    )

    parser.add_argument(
        "--topk_abs",
        type=int,
        default=-1,
        help="Top k for the absolute energy loss. Default is -1 (use all).",
    )

    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Path to load the model. Default is empty.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save the model. Default is empty.",
    )

    parser.add_argument(
        "--if_resume",
        type=str2bool,
        default=False,
        help="Whether to resume the training. Default is False.",
    )

    # for testing
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=-1,
        help="Epoch for loading the model. Default is -1.",
    )

    parser.add_argument(
        "--if_continue",
        type=str2bool,
        default=False,
        help="Weather to continue the test or generate data. Default is False.",
    )

    parser.add_argument(
        "--max_cycle",
        type=int,
        default=250,
        help="Maximum number of SCF cycles. Default is 250 and -1 for no iteration.",
    )

    parser.add_argument(
        "--if_rotate",
        type=str2bool,
        default=False,
        help="Weather to use rotation. Default is False.",
    )

    parser.add_argument(
        "--if_rotate_random",
        type=str2bool,
        default=False,
        help="Weather to use rotation. Default is False.",
    )

    parser.add_argument(
        "--max_memory_gpu",
        type=int,
        default=4000,
        help="Maximum memory for GPU calculation in MB. Default is 4000.",
    )

    args = parser.parse_args()

    args.name_mol_input = args.name_mol.copy()
    args.name_mol = gen_name_args(args.name_mol, args.dataset, args.name_mol_reverse)

    if args.activation_memory_budget is not None:
        print(
            "Warning: --activation_memory_budget is currently unused and kept only "
            "for backward compatibility.",
            flush=True,
        )

    print("Arguments:", flush=True)
    for arg_ in vars(args):
        print(f"{arg_}: {getattr(args, arg_)}", flush=True)
    print("", flush=True)

    return args
