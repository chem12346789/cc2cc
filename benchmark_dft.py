"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product

from cc2cc.benchmark_rks import benchmark_rks
from cc2cc.benchmark_uks import benchmark_uks
from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils import DataRecord
from cc2cc.utils import MAIN_PATH

if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    print_computer_info(args.device)

    # 1. Test loop
    if len(args.name_mol_input) == 1:
        data_record = DataRecord(
            MAIN_PATH
            / f"validate/ccdft_{args.basis}_{args.load}_{args.dataset}_{args.name_mol_input[0]}.csv",
            if_continue=args.if_continue,
        )
    else:
        data_record = DataRecord(
            MAIN_PATH / f"validate/ccdft_{args.basis}_{args.load}_{args.dataset}.csv",
            if_continue=args.if_continue,
        )
    error_molecule = []

    for name_mol in args.name_mol:
        name = f"{name_mol}_{args.basis}"
        mol = gen_mole(
            name_mol,
            args.basis,
            ma_basis=True,
            dataset_name=args.dataset,
        )

        if mol is None:
            print(f"SKIP: {name}")
            continue

        if args.if_continue:
            if ("name" in data_record.df_dict.keys()) and (
                name in data_record.df_dict["name"]
            ):
                print(f"SKIP: {name}")
                continue

        try:
            if mol.spin == 0:
                benchmark_rks(mol, name, data_record)
            else:
                benchmark_uks(mol, name, data_record)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {name_mol}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol}")
        print()

    print(f"Error molecule: {error_molecule}")
