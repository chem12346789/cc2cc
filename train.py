import argparse
import json
from pathlib import Path

from cc2cc.utils import add_args
from cc2cc.utils.parser import gen_name_args
from cc2cc.train_model import train_model

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_DEFAULT_SPLIT_CANDIDATES = (
    "dataset_split.json",
    "mol1.json",
    "dataset_split_mol1.json",
    "test.json",
)


def _default_split_path():
    for name in _DEFAULT_SPLIT_CANDIDATES:
        path = _CONFIG_DIR / name
        if path.exists():
            return path
    return _CONFIG_DIR / _DEFAULT_SPLIT_CANDIDATES[0]


DEFAULT_SPLIT_PATH = _default_split_path()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    parser.add_argument(
        "--split_config",
        type=str,
        default=str(DEFAULT_SPLIT_PATH),
        help="Path to JSON file defining train/eval splits.",
    )
    args = add_args(parser)

    split_path = Path(args.split_config)
    if not split_path.is_absolute():
        split_path = (Path(__file__).resolve().parent / split_path).resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"Split configuration not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        split_config = json.load(f)

    def _config_list(key):
        value = split_config.get(key, [])
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError(f"'{key}' in {split_path} must be a list, got {type(value)}")
        return value

    train_str_list = _config_list("train")
    eval_str_list = _config_list("eval")
    train_str_exclude_list = _config_list("train_exclude")
    eval_str_exclude_list = _config_list("eval_exclude")

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
            f"Train/eval overlap detected in {split_path} (count={len(overlap)}): "
            f"{preview}{suffix}"
        )

    print(f"Train set size: {len(train_str_list)}")
    print(f"Train set: {train_str_list}")
    print(f"Eval set size: {len(eval_str_list)}")
    print(f"Eval set: {eval_str_list}")
    train_model(train_str_list, eval_str_list, args)
