import argparse
import numpy as np


def gen_distance_list(distance_list):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=100,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
    )
    parser.add_argument(
        "--load_number",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--molecular_list",
        nargs="+",
        type=str,
        default=[
            "methane",
            "ethane",
            "ethylene",
            "acetylene",
        ],
        help="Name of molecular.",
    )
    parser.add_argument(
        "--distance_list",
        nargs="+",
        type=float,
        help="Distance between atom H to the origin. Default is 1.0.",
        default=[0.0],
    )
    parser.add_argument(
        "--extend_atom",
        type=str,
        nargs="+",
        default=["0-1"],
        help="Number of atoms to extend. Default is 0.",
    )
    parser.add_argument(
        "--extend_xyz",
        type=int,
        nargs="+",
        default=[0],
        help="Number of xyz to extend. 0 for x, 1 for y, 2 for z. Default is 0.",
    )
    args = parser.parse_args()
    for i in range(len(args.extend_xyz)):
        args.extend_xyz[i] += 1
    args.distance_list = gen_distance_list(args.distance_list)

    return args
