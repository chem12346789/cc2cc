"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product

from cc2cc import add_args, test_rks, test_uks
from cc2cc.utils import gen_mole, print_gpu_info
from cc2cc.utils import Grid, ModelClass, DataRecord
from cc2cc.utils import MAIN_PATH

if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    print_gpu_info(args.device)

    # 1. Init the model
    modeldict = ModelClass(args)
    if "test" not in args.load:
        modeldict.init_model()
        modeldict.eval()

    # 2. Test loop
    if args.disp is None:
        file_prefix = f"validate/ccdft_{args.basis}_{args.load}"
    else:
        file_prefix = f"validate/ccdft_{args.basis}_{args.load}_{args.disp}"
    if len(args.name_mol_input) == 1:
        data_record = DataRecord(
            MAIN_PATH / f"{file_prefix}_{args.dataset}_{args.name_mol_input[0]}.csv",
            if_continue=args.if_continue,
        )
    else:
        data_record = DataRecord(
            MAIN_PATH / f"{file_prefix}_{args.dataset}.csv",
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
            args.if_basis_str,
            args.dataset,
        )

        if mol is None:
            print(f"SKIP: {name}")
            continue

        if args.n_rad is not None and args.n_ang is not None:
            name = f"{name}_{args.n_rad}_{args.n_ang}"
        else:
            name = f"{name}_default"

        if args.if_continue:
            if ("name" in data_record.df_dict.keys()) and (
                name in data_record.df_dict["name"]
            ):
                print(f"SKIP: {name}")
                continue

        grids = Grid(mol, n_rad=args.n_rad, n_ang=args.n_ang)

        # try:
        if mol.spin == 0:
            test_rks(
                mol,
                grids,
                name,
                modeldict,
                data_record,
                args,
            )
        else:
            test_uks(
                mol,
                grids,
                name,
                modeldict,
                data_record,
                args,
            )
        # except (ValueError, RuntimeError) as e:
        #     print(f"ERROR: {name_mol} {extend_atom} {extend_xyz} {distance}")
        #     print(e)
        #     error_molecule.append(name)
        #     print(f"Error molecule: {error_molecule}")
        # finally:
        #     print(f"Processed: {name_mol} {extend_atom} {extend_xyz} {distance}")
        print()

    print(f"Error molecule: {error_molecule}")
