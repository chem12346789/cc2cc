import os
import argparse

from cc2cc import train_model
from cc2cc.utils import add_args
from cc2cc.utils.parser import gen_name_args

train_str_dict = [
    "molecule0",
    # "molecule1",
    # "molecule2",
    # "molecule3",
]
eval_str_dict = [
    # "molecule4",
    "molecule5",
]

# train_str_dict = [
#     "molecule0",
#     "molecule1",
#     "molecule2",
#     "molecule3_in_w411",
# ]
# eval_str_dict = [
#     "molecule4_in_w411",
#     "molecule5_in_w411",
# ]

if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    train_str_dict = gen_name_args(train_str_dict, args)
    eval_str_dict = gen_name_args(eval_str_dict, args)
    train_model(train_str_dict, eval_str_dict, args)
