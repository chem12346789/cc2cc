"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
import torch

from cc2cc.utils import (
    MAIN_PATH,
    print_computer_info,
    config_list,
    process_config,
    add_args,
    gen_mole,
    print_computer_info,
)
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.parser import gen_name_args


def main():
    parser = argparse.ArgumentParser(
        description="Test the model or benchmark DFT calculations. Other parameters are from the argparse."
    )
    parser.add_argument(
        "--benchmark_method",
        type=str,
        nargs="+",
        default=["b3lyp"],
        help="Benchmark method for DFT calculations. Default is b3lyp.",
    )
    parser.add_argument(
        "--benchmark_disp",
        type=str,
        nargs="+",
        default=None,
        help="Dispersion correction for benchmark DFT calculations. Default is None.",
    )
    parser.add_argument(
        "--split_config",
        type=str,
        default="mini.json",
        help="Path to JSON file defining train/eval splits.",
    )
    args = add_args(parser)

    print_computer_info(args.device)

    split_config = process_config(args.split_config)
    train_list = config_list(split_config, "train")
    eval_list = config_list(split_config, "eval")
    train_exclude_list = config_list(split_config, "train_exclude")
    eval_exclude_list = config_list(split_config, "eval_exclude")

    train_list = gen_name_args(train_list, args.dataset, args.name_mol_reverse)
    train_exclude_list = gen_name_args(
        train_exclude_list, args.dataset, args.name_mol_reverse, if_exclude=True
    )
    eval_list = gen_name_args(eval_list, args.dataset, args.name_mol_reverse)
    eval_exclude_list = gen_name_args(
        eval_exclude_list, args.dataset, args.name_mol_reverse, if_exclude=True
    )

    train_list = [mol for mol in train_list if mol not in train_exclude_list]
    eval_list = [mol for mol in eval_list if mol not in eval_exclude_list]

    modeldict = ModelClass(args)
    modeldict.init_model(init_train=False)
    modeldict.init_database(train_list, eval_list)

    for batch in modeldict.database_train.data_gpu:
        batch = modeldict.database_train.process_batch(
            batch, device=modeldict.local_rank
        )
        input_ = batch["input"]
        weight = batch["weight"]
        sum_target = batch["energy_target"]
        data_record = {"name": batch["name"]}
        input_, output = modeldict.model_output(input_, weight)
        sum_output = torch.sum(output)
        data_record["loss_ene"] = torch.abs(sum_target - sum_output).detach().to("cpu")
        print(data_record)


if __name__ == "__main__":
    main()
