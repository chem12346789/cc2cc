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
    argument_specs = [
        # Data generation
        (
            ("--name_mol", "-m"),
            dict(
                nargs="+",
                type=str,
                default=[],
                help="Name of molecule. Default is None (all the dataset).",
            ),
        ),
        (
            ("--name_mol_reverse",),
            dict(
                type=str2bool,
                default=False,
                help="Whether to reverse the order of the molecule names. Default is False.",
            ),
        ),
        (
            ("--md_number",),
            dict(
                type=int,
                default=0,
                help="MD frame number to generate the data. Default is 0.",
            ),
        ),
        (
            ("--mp_number",),
            dict(type=int, default=0, help="Number of training cycles. Default is 1."),
        ),
        (
            ("--mp_total",),
            dict(
                type=int,
                default=3,
                help="Total number of training cycles. Default is 3.",
            ),
        ),
        (
            ("--if_eval",),
            dict(
                type=str2bool,
                default=False,
                help="Whether to use the evaluation mode in generating the data. Default is False.",
            ),
        ),
        (
            ("--grid_level",),
            dict(type=int, default=4, help="Grid level for the calculation."),
        ),
        (
            ("--basis",),
            dict(type=str, default="cc-pVDZ", help="Basis set for the calculation."),
        ),
        (
            ("--dataset",),
            dict(
                type=str,
                default="mol",
                help="Name of the dataset. Default is mol (training and testing).",
            ),
        ),
        (
            ("--check_convergence",),
            dict(
                type=str2bool,
                default=True,
                help="Whether to check the convergence of the wave function. Default is True.",
            ),
        ),
        # Training
        (
            ("--model",),
            dict(
                type=str,
                default="transformer+dense_mix_e3nn_4",
                help="Model for training.",
            ),
        ),
        (
            ("--load",),
            dict(
                type=str,
                default="",
                help="Path to load the model. Default is empty.",
            ),
        ),
        # Testing
        (
            ("--load_epoch",),
            dict(
                type=int, default=-1, help="Epoch for loading the model. Default is -1."
            ),
        ),
        (
            ("--if_continue",),
            dict(
                type=str2bool,
                default=False,
                help="Weather to continue the test or generate data. Default is False.",
            ),
        ),
        (
            ("--max_cycle",),
            dict(
                type=int,
                default=250,
                help="Maximum number of SCF cycles. Default is 250 and -1 for no iteration.",
            ),
        ),
        (
            ("--if_rotate",),
            dict(
                type=str2bool,
                default=False,
                help="Weather to use rotation. Default is False.",
            ),
        ),
        (
            ("--if_rotate_random",),
            dict(
                type=str2bool,
                default=False,
                help="Weather to use rotation. Default is False.",
            ),
        ),
        (
            ("--max_memory_gpu",),
            dict(
                type=int,
                default=4000,
                help="Maximum memory for GPU calculation in MB. Default is 4000.",
            ),
        ),
    ]

    for flags, kwargs in argument_specs:
        parser.add_argument(*flags, **kwargs)

    args = parser.parse_args()

    args.name_mol_input = args.name_mol.copy()
    args.name_mol = gen_name_args(args.name_mol, args.dataset, args.name_mol_reverse)

    print("Arguments:", flush=True)
    for arg_ in vars(args):
        print(f"{arg_}: {getattr(args, arg_)}", flush=True)
    print("", flush=True)

    return args
