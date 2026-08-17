"""
@package docstring
Documentation for this module.

More details.
"""

import argparse
import json
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def config_list(config, key):
    value = config.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"'{key}' in {config} must be a list, got {type(value)}")
    return value


def process_config(split_config):
    split_path = Path(split_config)
    if not split_path.is_absolute():
        split_path = (_CONFIG_DIR / split_path).resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"Split configuration not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        split_config = json.load(f)

    return split_config


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

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for the training. Default is cuda.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Precision for the training. Default is float64.",
    )
    parser.add_argument(
        "--name_mol",
        "-m",
        nargs="+",
        type=str,
        default=[],
        help="Name of molecule. Default is None (all the dataset).",
    )
    parser.add_argument(
        "--md_number",
        type=int,
        default=0,
        help="MD frame number to generate the data. Default is 0.",
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
    # model loading and saving
    parser.add_argument(
        "--model",
        type=str,
        default="transformer+dense_mix_e3nn_4",
        help="Model for training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Path to load the model. Default is empty.",
    )
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=-1,
        help="Epoch for loading the model. Default is -1.",
    )
    # behavior of the program
    parser.add_argument(
        "--if_continue",
        type=str2bool,
        default=False,
        help="Whether to continue the test or generate data. Default is False.",
    )
    parser.add_argument(
        "--name_mol_reverse",
        type=str2bool,
        default=False,
        help="Whether to reverse the order of the molecule names. Default is False.",
    )

    args = parser.parse_args()

    args.name_mol_input = args.name_mol.copy()
    args.name_mol = gen_name_args(args.name_mol, args.dataset, args.name_mol_reverse)

    print("Arguments:", flush=True)
    for arg_ in vars(args):
        print(f"{arg_}: {getattr(args, arg_)}", flush=True)
    print("", flush=True)

    return args
