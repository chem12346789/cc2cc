"""
@package docstring
Documentation for this module.

More details.
"""

import argparse
import numpy as np

from cc2cc.utils.mol import dataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True" "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False" "f", "n", "0"):
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
        default="",
        help="Name of molecular. Default is empty (all the dataset).",
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
        "--basis",
        type=str,
        default="cc-pVDZ",
        help="Basis set for the calculation.",
    )

    parser.add_argument(
        "--if_basis_str",
        type=str2bool,
        default=True,
        help="Weather to use the basis set from basissetexchange. See https://www.basissetexchange.org. Default is False.",
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
        default="False",
        help="Weather to use the noniterative CCSD(T) in the coupled cluster method. Default is False.",
    )

    # for machine learning

    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Weather to load the saved check point. Default is empty.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory for saving the model. Default is empty.",
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
        default=1000000,
        help="Batch size for training. Default is 64 (FCnet).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Precision for the training. Default is float64.",
    )

    parser.add_argument(
        "--with_eval",
        type=str2bool,
        default=True,
        help="Weather to use the reduce on plateau for the learning rate. Default is True. This will use the data from the eval set.",
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

    args = parser.parse_args()
    for i in range(len(args.extend_xyz)):
        args.extend_xyz[i] += 1

    args.distance_list = gen_logger(args.distance_list)

    if len(args.name_mol) == 0:
        args.name_mol = dataset[args.dataset]["molecular"]

    return args
