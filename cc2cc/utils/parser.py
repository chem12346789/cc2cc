"""
@package docstring
Documentation for this module.

More details.
"""

import argparse
import numpy as np

from cc2cc.utils.mol import dataset

periodic_table = {
    -1: "all",
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
}


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


def gen_logger(distance_list):
    """
    Function to distance list and generate logger
    """
    if len(distance_list) == 3:
        distance_l = np.linspace(
            distance_list[0], distance_list[1], int(distance_list[2])
        )
    else:
        distance_l = distance_list
    return distance_l


def gen_name_args(name_args, args):
    """
    Function to generate name args
    """
    if name_args is None:
        name_mol_new = dataset[args.dataset]["molecule"]
    elif len(name_args) == 0:
        name_mol_new = []
    else:
        name_mol_new = []
        for i in range(len(name_args)):
            if name_args[i].startswith("molecule"):
                if name_args[i] in dataset[args.dataset].keys():
                    name_mol_new += dataset[args.dataset][name_args[i]]
                else:
                    raise ValueError(
                        f"Invalid molecule name: {name_args[i]}. "
                        f"Please use a valid molecule name in {dataset[args.dataset].keys()}"
                    )
            else:
                name_mol_new.extend([name_args[i]])
    print(f"Name of molecule: {name_mol_new}")
    return name_mol_new


def add_args(parser: argparse.ArgumentParser):
    """
    Documentation for a function.

    More details.
    """
    parser.add_argument(
        "--name_mol",
        "-m",
        nargs="+",
        type=str,
        default=None,
        help="Name of molecule. Default is None (all the dataset).",
    )

    parser.add_argument(
        "--distance_list",
        "-dl",
        nargs="+",
        type=float,
        help="Distance between atom H to the origin. Default is 1.0.",
        default=1.0,
    )

    parser.add_argument(
        "--extend_atom",
        type=str,
        nargs="+",
        default=0,
        help="Number of atoms to extend. Default is 0.",
    )

    parser.add_argument(
        "--extend_xyz",
        type=int,
        nargs="+",
        default=0,
        help="Number of xyz to extend. 0 for x, 1 for y, 2 for z. Default is 0.",
    )

    parser.add_argument(
        "--n_rad",
        type=int,
        default=None,
        help="Number of radial points. Default is None.",
    )

    parser.add_argument(
        "--n_ang",
        type=int,
        default=None,
        help="Number of angular points. Default is None.",
    )

    parser.add_argument(
        "--basis",
        type=str,
        default="cc-pVDZ",
        help="Basis set for the calculation.",
    )

    parser.add_argument(
        "--if_basis_str",
        type=str2bool,
        default=True,
        help="Whether to use the basis set from basissetexchange. "
        "See https://www.basissetexchange.org. Default is True.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mol",
        help="Name of the dataset. Default is mol (training and testing).",
    )

    parser.add_argument(
        "--cc_triple",
        type=str2bool,
        default=False,
        help="Whether to use the noniterative CCSD(T) in the coupled cluster method. "
        "Default is False.",
    )

    parser.add_argument(
        "--if_grad",
        type=str2bool,
        default=False,
        help="Whether to calculate the gradient. Default is False.",
    )

    # for machine learning
    parser.add_argument(
        "--model",
        type=str,
        default="densenet",
        help="Model for the training. Default is densenet.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=10000,
        help="Number of epoch for training. Default is 10000.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training. Default is 1 (molecule / batch).",
    )

    parser.add_argument(
        "--if_load_to_gpu_once",
        type=str2bool,
        default=True,
        help="Whether to load all the data to GPU once. Default is True. "
        "This will use more memory, but faster.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Precision for the training. Default is float64.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the training. Default is 1e-4.",
    )

    parser.add_argument(
        "--iters_to_accumulate",
        type=int,
        default=1,
        help="Number of iterations to accumulate the gradient. Default is 1.",
    )

    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="Max norm for the gradient. Default is 1.0.",
    )

    parser.add_argument(
        "--with_eval",
        type=str2bool,
        default=True,
        help="Whether to use the reduce on plateau. Default is True. \n"
        "This will use the data from the eval set.",
    )

    parser.add_argument(
        "--eval_step",
        type=int,
        default=1,
        help="Step for evaluation. Default is 1.",
    )

    parser.add_argument(
        "--loss_multiplier",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )

    parser.add_argument(
        "--train_atom",
        type=int,
        default=1,
        help="Atom for training. Default is 1 (1 for Hydrogen).",
    )

    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Whether to load the saved check point. Default is empty.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory for saving the model. Default is empty.",
    )

    # for testing
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=-1,
        help="Epoch for loading the model. Default is -1.",
    )

    parser.add_argument(
        "--density_restriction",
        type=float,
        default=0.0,
        help="Lambda for the density restriction. Default is 0.0.",
    )

    parser.add_argument(
        "--if_continue",
        type=str2bool,
        default=False,
        help="Weather to continue the data record. Default is False.",
    )

    args = parser.parse_args()
    for i in range(len(args.extend_xyz)):
        args.extend_xyz[i] += 1

    args.distance_list = gen_logger(args.distance_list)
    args.name_mol = gen_name_args(args.name_mol, args)

    if args.train_atom not in periodic_table.keys():
        raise ValueError(
            f"Invalid train_atom value: {args.train_atom}. Please use a valid atomic number."
        )
    else:
        args.train_atom = periodic_table[args.train_atom]

    print("Arguments:")
    print(f"Name of molecule: {args.name_mol}")
    print(f"Distance list: {args.distance_list}")
    print(f"Extend atom: {args.extend_atom}")
    print(f"Extend xyz: {args.extend_xyz}")
    print(f"Number of radial points: {args.n_rad}")
    print(f"Number of angular points: {args.n_ang}")
    print(f"Basis set: {args.basis}")
    print(f"Use basis set from basissetexchange: {args.if_basis_str}")
    print(f"Dataset: {args.dataset}")
    print(f"CCSD(T): {args.cc_triple}")
    print(f"Gradient: {args.if_grad}")
    print(f"Model: {args.model}")
    print(f"Epoch: {args.epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Load to GPU once: {args.if_load_to_gpu_once}")
    print(f"Precision: {args.precision}")
    print(f"Learning rate: {args.lr}")
    print(f"Iterations to accumulate: {args.iters_to_accumulate}")
    print(f"Max norm: {args.max_norm}")
    print(f"With eval: {args.with_eval}")
    print(f"Eval step: {args.eval_step}")
    print(f"Loss multiplier: {args.loss_multiplier}")
    print(f"Train atom: {args.train_atom}")
    print(f"Load: {args.load}")
    print(f"Save directory: {args.save_dir}")
    print(f"Load epoch: {args.load_epoch}")
    print(f"Density restriction: {args.density_restriction}")
    print(f"Continue: {args.if_continue}")

    return args
