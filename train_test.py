"""
Train the model for the small dataset.
Used to test the code and debug the model.
"""

import argparse

from cc2cc import train_model
from cc2cc.utils import add_args, print_gpu_info
from cc2cc.utils.parser import gen_name_args

train_str_list = [
    "W4_11-si",
    "W4_11-f",
    "W4_11-c",
]
eval_str_list = [
    "W4_11-cf4",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    print_gpu_info(args.device)

    train_str_list = gen_name_args(train_str_list, args)
    eval_str_list = gen_name_args(eval_str_list, args)
    train_model(train_str_list, eval_str_list, args)
