"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product

from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils import Grid, ModelClass, DataRecord
from cc2cc.utils import MAIN_PATH
from cc2cc.test_model_rks import test_model_rks
from cc2cc.test_model_uks import test_model_uks

if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    print_computer_info(args.device)

    # 1. Init the model
    modeldict = ModelClass(args)
    if "test" not in args.load:
        modeldict.init_model(if_validate=True)
        modeldict.eval()

    # 2. Test loop
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

    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        args.name_mol,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
    ):
        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        mol = gen_mole(
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
            args.basis,
            args.dataset,
            if_rotate=args.if_rotate,
            if_rotate_random=args.if_rotate_random,
        )

        if mol is None:
            print(f"SKIP: {name}")
            continue

        if args.if_continue:
            if ("name" in data_record.df_dict) and (
                name in data_record.df_dict["name"]
            ):
                print(f"SKIP: {name}")
                continue

        grids = Grid(mol, args.grid_level)

        try:
            if mol.spin == 0:
                test_model_rks(
                    mol,
                    grids,
                    name,
                    modeldict,
                    data_record,
                    args,
                )
            else:
                test_model_uks(
                    mol,
                    grids,
                    name,
                    modeldict,
                    data_record,
                    args,
                )
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {name_mol} {extend_atom} {extend_xyz} {distance}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol} {extend_atom} {extend_xyz} {distance}")
        print()

    print(f"Error molecule: {error_molecule}")
