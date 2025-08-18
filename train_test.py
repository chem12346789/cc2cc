"""
Train the model for the small dataset.
Used to test the code and debug the model.
"""

import argparse

from cc2cc import train_model
from cc2cc.utils import add_args, print_computer_info
from cc2cc.utils.parser import gen_name_args

train_str_list = [
    "molecule0",
    "molecule1-W4_11",
    "molecule2-W4_11",
    "molecule3-W4_11",
    "molecule4-W4_11",
    "BH9-08_9R2",  # 5
]
train_str_exclude_list = [
    "W4_11-propane",  # 3
    "molecule1-ACC24",
    "molecule1-GAPS",
    "molecule1-GW100",
    "molecule1-MRADC",
    "molecule1-S30L",
    "molecule2-ACC24",
    "molecule2-GAPS",
    "molecule2-GW100",
    "molecule2-MRADC",
    "molecule2-S30L",
    "molecule3-ACC24",
    "molecule3-GAPS",
    "molecule3-GW100",
    "molecule3-MRADC",
    "molecule3-S30L",
    "molecule4-ACC24",
    "molecule4-GAPS",
    "molecule4-GW100",
    "molecule4-MRADC",
    "molecule4-S30L",
]
eval_str_list = [
    "W4_11-sif",
    "W4_11-cf4",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    print_computer_info(args.device)

    train_str_list = gen_name_args(train_str_list, args.dataset, args.name_mol_reverse)
    eval_str_list = gen_name_args(eval_str_list, args.dataset, args.name_mol_reverse)
    train_model(train_str_list, eval_str_list, args)
