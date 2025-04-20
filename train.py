import argparse

from cc2cc import train_model
from cc2cc.utils import add_args, print_gpu_info
from cc2cc.utils.parser import gen_name_args

train_str_dict = [
    "molecule0",
    "molecule1",
    "molecule2-W4_11",
    "molecule2-G2RC",
    "molecule2-BSR36",
    "molecule2-ALK8",
    "molecule2-HEAVYSB11",
    "molecule3-W4_11",
    "molecule3-ALK8",
    "molecule3-HEAVYSB11",
    "molecule4-ALK8",
    "ADIM6-AD2",  # 4
    # "molecule5-BSR36",
    # "molecule5-ALK8",
    # "molecule5-BSR36",
]
eval_str_dict = [
    "molecule4-W4_11",
    "molecule5-W4_11",
    "molecule6-BSR36",
    "ADIM6-AD3",  # 6
    "molecule7-MB16_43",
    "molecule7-BSR36",
    "molecule8-MB16_43",
    "ADIM6-AD4",  # 8
    "molecule8-BSR36",
    "molecule8-IDISP",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    print_gpu_info(args.device)

    train_str_dict = gen_name_args(train_str_dict, args)
    eval_str_dict = gen_name_args(eval_str_dict, args)
    train_model(train_str_dict, eval_str_dict, args)
