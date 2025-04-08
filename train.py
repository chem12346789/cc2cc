import argparse

from cc2cc import train_model
from cc2cc.utils import add_args, print_gpu_info
from cc2cc.utils.parser import gen_name_args

train_str_dict = [
    "molecule0",
    "molecule1",
    "molecule2-W4_11",
    "molecule3-W4_11",
    "molecule4-W4_11",
]
eval_str_dict = [
    "molecule5-W4_11",
    "molecule5-BH9",
    "molecule5-BH76",
]

if __name__ == "__main__":
    print_gpu_info()
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    train_str_dict = gen_name_args(train_str_dict, args)
    eval_str_dict = gen_name_args(eval_str_dict, args)
    train_model(train_str_dict, eval_str_dict, args)
