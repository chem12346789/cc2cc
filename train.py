import argparse

from cc2cc import train_model
from cc2cc.utils import add_args, print_computer_info
from cc2cc.utils.parser import gen_name_args

train_str_list = [
    "molecule0",
    "molecule1",
    # "molecule2",
    # # "molecule3-ALK8",
    # "molecule3-ALK8",
    # "molecule3-HEAVYSB11",
    # "molecule3-W4_11",
    # "molecule3-AL2X6",
    # "molecule4-ALK8",
    # "molecule4-W4_11",
    # "BH9-08_9R2",  # 5
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
    "W4_11-propane",  # 3
    "ADIM6-AD2",  # 4
    "molecule5-ALK8",
    "molecule5-BSR36",
    "molecule5-W4_11",
    "molecule6-BSR36",
    "ADIM6-AD3",  # 6
    # "molecule7-MB16_43",
    # "molecule7-BSR36",
    # "molecule8-MB16_43",
    # "ADIM6-AD4",  # 8
    # "molecule8-BSR36",
    # "molecule8-IDISP",
]
eval_str_exclude_list = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    print_computer_info(args.device)

    train_str_list = gen_name_args(train_str_list, args)
    train_str_exclude_list = gen_name_args(
        train_str_exclude_list, args, if_exclude=True
    )
    eval_str_list = gen_name_args(eval_str_list, args)
    eval_str_exclude_list = gen_name_args(eval_str_exclude_list, args, if_exclude=True)

    # remove the same name in train and train_str_exclude_list
    train_str_list = [
        mol for mol in train_str_list if mol not in train_str_exclude_list
    ]

    # remove the same name in eval and eval_str_exclude_list
    eval_str_list = [mol for mol in eval_str_list if mol not in eval_str_exclude_list]

    train_model(train_str_list, eval_str_list, args)
