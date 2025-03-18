"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product
import os

from cc2cc import add_args, test_rks, test_uks
from cc2cc.utils import gen_mole
from cc2cc.utils import Grid, ModelDict, DataRecord
from cc2cc.utils import MAIN_PATH


# from cadft.utils.ModelDict_xy import ModelDict
# from cadft.utils import ModelDict_xy1 as ModelDict
# from cadft.utils.ModelDict_xy2 import ModelDict

# class ModelDict_data()
if __name__ == "__main__":
    # 0. Prepare the args
    print(f"PID: {os.getpid()}")
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    # 1. Init the model
    modeldict = ModelDict(args)
    modeldict.load_model()
    modeldict.eval()

    # 2. Test loop
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
            if name in data_record.df_dict["name"]:
                print(f"SKIP: {name}")
                continue

        grids = Grid(mol, n_rad=args.n_rad, n_ang=args.n_ang)

        try:
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
        except ValueError as e:
            print(f"ERROR: {name_mol} {extend_atom} {extend_xyz} {distance}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol} {extend_atom} {extend_xyz} {distance}")
        print()

    print(f"Error molecule: {error_molecule}")
