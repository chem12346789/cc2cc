import argparse

from cc2cc.utils import add_args, print_computer_info, config_list, process_config
from cc2cc.utils.parser import gen_name_args
from cc2cc.train_model import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
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
    train_str_list = config_list(split_config, "train")
    eval_str_list = config_list(split_config, "eval")
    train_str_exclude_list = config_list(split_config, "train_exclude")
    eval_str_exclude_list = config_list(split_config, "eval_exclude")

    train_str_list = gen_name_args(train_str_list, args.dataset, args.name_mol_reverse)
    train_str_exclude_list = gen_name_args(
        train_str_exclude_list, args.dataset, args.name_mol_reverse, if_exclude=True
    )
    eval_str_list = gen_name_args(eval_str_list, args.dataset, args.name_mol_reverse)
    eval_str_exclude_list = gen_name_args(
        eval_str_exclude_list, args.dataset, args.name_mol_reverse, if_exclude=True
    )

    # remove the same name in train and train_str_exclude_list
    train_str_list = [
        mol for mol in train_str_list if mol not in train_str_exclude_list
    ]

    # remove the same name in eval and eval_str_exclude_list
    eval_str_list = [mol for mol in eval_str_list if mol not in eval_str_exclude_list]

    overlap = sorted(set(train_str_list) & set(eval_str_list))
    if overlap:
        preview = ", ".join(overlap[:8])
        suffix = " ..." if len(overlap) > 8 else ""
        raise ValueError(
            f"Train/eval overlap detected in {args.split_config} (count={len(overlap)}): "
            f"{preview}{suffix}"
        )

    print(f"Train set size: {len(train_str_list)}")
    print(f"Train set: {train_str_list}")
    print(f"Eval set size: {len(eval_str_list)}")
    print(f"Eval set: {eval_str_list}")
    train_model(train_str_list, eval_str_list, args)
