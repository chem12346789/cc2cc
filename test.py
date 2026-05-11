"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from pathlib import Path

from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils import ModelClass, DataRecord
from cc2cc.utils import MAIN_PATH
from cc2cc.test_model_rks import test_model_rks
from cc2cc.test_model_uks import test_model_uks


def _build_record_path(args) -> Path:
    base = f"ccdft_{args.basis}_{args.load}_{args.dataset}"
    if len(args.name_mol_input) == 1:
        base = f"{base}_{args.name_mol_input[0]}"
    return MAIN_PATH / f"validate/{base}.csv"


def _should_skip(name: str, data_record, args) -> bool:
    if not args.if_continue:
        return False
    return ("name" in data_record.df_dict) and (name in data_record.df_dict["name"])


def _run_one(mol, name, modeldict, data_record, args):
    if mol.spin == 0:
        test_model_rks(mol, name, modeldict, data_record, args)
    else:
        test_model_uks(mol, name, modeldict, data_record, args)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    print_computer_info(args.device)

    modeldict = ModelClass(args)
    if "test" not in args.load:
        modeldict.init_model(if_validate=True)
        modeldict.eval()

    data_record = DataRecord(
        _build_record_path(args),
        if_continue=args.if_continue,
    )
    error_molecule = []

    for name_mol in args.name_mol:
        name = f"{name_mol}_{args.basis}"
        mol = gen_mole(
            name_mol,
            args.basis,
            dataset_name=args.dataset,
            if_rotate=args.if_rotate,
            if_rotate_random=args.if_rotate_random,
        )

        if mol is None:
            print(f"SKIP: {name}")
            continue

        if _should_skip(name, data_record, args):
            print(f"SKIP: {name}")
            continue

        _run_one(mol, name, modeldict, data_record, args)

        # try:
        #     _run_one(mol, name, modeldict, data_record, args)
        # except (ValueError, RuntimeError) as e:
        #     print(f"ERROR: {name_mol}")
        #     print(e)
        #     error_molecule.append(name)
        #     print(f"Error molecule: {error_molecule}")
        # finally:
        #     print(f"Processed: {name_mol}")

    print(f"Error molecule: {error_molecule}")


if __name__ == "__main__":
    main()
