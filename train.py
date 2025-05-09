import argparse

from cc2cc import train_model
from cc2cc.utils import add_args, print_gpu_info
from cc2cc.utils.parser import gen_name_args

train_str_list = [
    "molecule0",
    "molecule1-AHB21",
    "molecule1-ALK8",
    "molecule1-ALKBDE10",
    "molecule1-AL2X6",
    "molecule1-BSR36",
    "molecule1-BH76",
    "molecule1-BH9",
    "molecule1-CARBHB12",
    "molecule1-CHB6",
    "molecule1-DC13",
    "molecule1-DIPCS10",
    "molecule1-FH51",
    "molecule1-G21EA",
    "molecule1-G21IP",
    "molecule1-G2RC",
    "molecule1-HAL59",
    "molecule1-HEAVY28",
    "molecule1-HEAVYSB11",
    "molecule1-IDISP",
    "molecule1-INV24",
    "molecule1-MB16_43",
    "molecule1-NBPRC",
    "molecule1-PA26",
    "molecule1-PNICO23",
    "molecule1-PX13",
    "molecule1-RC21",
    "molecule1-RG18",
    "molecule1-RSE43",
    "molecule1-S22",
    "molecule1-S66",
    "molecule1-SIE4x4",
    "molecule1-W4_11",
    "molecule1-WATER27",
    "molecule1-WCPT18",
    "molecule1-YBDE18",
    "molecule2-ALK8",
    "molecule2-BSR36",
    "molecule2-G2RC",
    "molecule2-HEAVYSB11",
    "molecule2-SIE4x4",
    "molecule2-W4_11",
    "molecule3-ALK8",
    "molecule3-HEAVYSB11",
    "molecule3-W4_11",
    "molecule4-ALK8",
    "molecule4-W4_11",
    "BH9-08_9R2",  # 5
]
train_str_exclude_list = [
    "W4_11-propane",  # 3
]
eval_str_list = [
    "W4_11-propane",  # 3
    "ADIM6-AD2",  # 4
    "molecule5-BSR36",
    "molecule5-ALK8",
    "molecule5-BSR36",
    "molecule5-W4_11",
    "molecule6-BSR36",
    "ADIM6-AD3",  # 6
    "molecule7-MB16_43",
    "molecule7-BSR36",
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

    print_gpu_info(args.device)

    train_str_list = gen_name_args(train_str_list, args)
    eval_str_list = gen_name_args(eval_str_list, args)

    # remove the same name in train and train_str_exclude_list
    train_str_list = list(set(train_str_list) - set(train_str_exclude_list))
    # remove the same name in eval and eval_str_exclude_list
    eval_str_list = list(set(eval_str_list) - set(eval_str_exclude_list))

    train_model(train_str_list, eval_str_list, args)
