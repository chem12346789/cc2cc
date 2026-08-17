import argparse

from cc2cc.utils import (
    add_args,
    print_computer_info,
    config_list,
    process_config,
)
from cc2cc.utils.parser import gen_name_args, str2bool
from cc2cc.train_model import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    parser.add_argument(
        "--rho_input",
        type=str,
        default="dft",
        choices=["dft", "dft_d3bj", "zmp"],
        help="Type of input density. ",
    )
    parser.add_argument(
        "--split_config",
        type=str,
        default="mini.json",
        help="Path to JSON file defining train/eval splits.",
    )
    parser.add_argument(
        "--append_mol0",
        type=int,
        default=0,
        help="Whether to append mol0 data. Default is 0 (do not append).",
    )
    parser.add_argument(
        "--mol0_weighting",
        type=int,
        default=1,
        help="Weighting scheme for atomic energy. Default is 1 (with 20 copies).",
    )
    parser.add_argument(
        "--grad_step",
        type=int,
        default=1,
        help="Interval for calculating the gradient. Default is 1 (calculate every step).",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["MSELoss", "L1Loss"],
        default="MSELoss",
        help="Loss function for the energy. Default is MSELoss. Other options are L1Loss.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10000,
        help="Number of epoch for training. Default is 10000.",
    )
    parser.add_argument(
        "--distributed",
        default=False,
        type=str2bool,
        help="Whether to use distributed training. Default is False.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the training. Default is 1e-4.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["AdamW", "Adafactor"],
        help="Optimizer for the training. Default is AdamW.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="constant",
        choices=["cosine", "constant", "cosine_warm"],
        help="Learning rate scheduler. Default is constant.",
    )
    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=1e-8,
        help="Minimum learning rate for cosine scheduler. Default is 1e-8.",
    )
    parser.add_argument(
        "--cosine_T",
        type=int,
        default=16,
        help="Number of periods for the cosine scheduler. Default is 16.",
    )
    parser.add_argument(
        "--cosine_T_mult",
        type=int,
        default=1,
        help="Multiplicative factor for the period of the cosine scheduler. Default is 1.",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="Max norm for the gradient. Default is 1.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the training. Default is None (no seed). ",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for the optimizer. Default is 1e-3.",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=1,
        help="Step for evaluation. Default is 1.",
    )
    parser.add_argument(
        "--if_relative_weight",
        type=str2bool,
        default=False,
        help="Whether to use relative weight for the loss function. Default is False.",
    )
    parser.add_argument(
        "--if_relative_weight_abs",
        type=str2bool,
        default=False,
        help="Whether to use relative weight for the absolute loss function. Default is False.",
    )
    parser.add_argument(
        "--relative_weight_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for the relative weight. Default is 1e-8.",
    )
    parser.add_argument(
        "--output_target",
        type=str,
        default="tol_delta_grids",
        choices=[
            "tol_delta_grids",
            "tol_delta_grids_l",
            "tol_delta_grids_l_erf",
            "exc_cc_grids",
            "b3lyp",
        ],
        help="Target for the output. Default is tol_delta_grids.",
    )
    parser.add_argument(
        "--loss_multiplier",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )
    parser.add_argument(
        "--loss_multiplier_abs",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )
    parser.add_argument(
        "--loss_multiplier_atomic",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )
    parser.add_argument(
        "--loss_multiplier_grad",
        type=float,
        default=1.0,
        help="Lambda for the loss function. Default is 1.0.",
    )
    parser.add_argument(
        "--if_grad",
        type=str2bool,
        default=False,
        help="Whether to calculate the gradient. Default is False.",
    )
    parser.add_argument(
        "--if_atomic",
        type=str2bool,
        default=False,
        help="Whether to calculate the atomic energy. Default is False.",
    )
    parser.add_argument(
        "--if_abs",
        type=str2bool,
        default=False,
        help="Whether to calculate the absolute energy. Default is False.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save the model. Default is empty.",
    )
    parser.add_argument(
        "--if_resume",
        type=str2bool,
        default=False,
        help="Whether to resume the training. Default is False.",
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

    # remove the same name in train and train_str_exclude_list
    train_list = [mol for mol in train_list if mol not in train_exclude_list]

    # remove the same name in eval and eval_str_exclude_list
    eval_list = [mol for mol in eval_list if mol not in eval_exclude_list]

    overlap = sorted(set(train_list) & set(eval_list))
    if overlap:
        preview = ", ".join(overlap[:8])
        suffix = " ..." if len(overlap) > 8 else ""
        raise ValueError(
            f"Train/eval overlap detected in {args.split_config} (count={len(overlap)}): "
            f"{preview}{suffix}"
        )

    print(f"Train set size: {len(train_list)}")
    print(f"Train set: {train_list}")
    print(f"Eval set size: {len(eval_list)}")
    print(f"Eval set: {eval_list}")
    train_model(train_list, eval_list, args)
