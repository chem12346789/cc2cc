import os
import argparse

from cc2cc import train_model
from cc2cc.utils import add_args
from cc2cc.utils.parser import gen_name_args

train_str_dict = [
    "molecule0",
    "W4_11-molecule1",
    "W4_11-molecule2",
    "W4_11-molecule3",
    "W4_11-molecule4",
]
eval_str_dict = [
    "W4_11-molecule5",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    train_str_dict = gen_name_args(train_str_dict, args)
    eval_str_dict = gen_name_args(eval_str_dict, args)
    train_model(train_str_dict, eval_str_dict, args)
