""" """

import argparse
from itertools import product

import numpy as np

from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils import Grid, DATA_PATH
from cc2cc.utils.parser import gen_name_args
from cc2cc.gen_cc import cc
from cc2cc.gen_ucc import ucc


train_str_list = [
    "molecule0-W4_11",
]

eval_str_list = [
    "molecule1-W4_11",
    "molecule2-W4_11",
    "molecule3-W4_11",
    "molecule4-W4_11",
    "molecule5-W4_11",
]

train_str_exclude_list = []
eval_str_exclude_list = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    error_molecule = []

    print_computer_info(args.device)

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

    if args.if_eval:
        name_mol_list, evaluate = eval_str_list[args.training_cycle :: 6], True
    else:
        name_mol_list, evaluate = train_str_list[args.training_cycle :: 3], False

    # name_mol_list = [
    #     "AHB21-1A",
    #     "AHB21-4A",
    #     "G21EA-EA_c-",
    #     "G21EA-EA_o-",
    #     "G21EA-EA_p-",
    #     "G21EA-EA_s-",
    #     "G21EA-EA_si-",
    # ]
    # evaluate = False

    error_molecule = []
    print(f"Name Molecule List: {name_mol_list}")

    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        name_mol_list,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
    ):
        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

        try:
            mol = gen_mole(
                name_mol,
                extend_atom,
                extend_xyz,
                distance,
                args.basis,
                ma_basis=False,
                dataset_name=args.dataset,
            )

            if mol is None:
                print(f"SKIP: {name_mol} {extend_atom} {extend_xyz} {distance}")
                continue

            grids = Grid(mol, args.grid_level)

            if args.if_continue:
                if (DATA_PATH / f"data_{name}.npz").exists():
                    continue
                    # if mol.charge >= 0:
                    #     print(f"SKIP: {name_mol} {extend_atom} {extend_xyz} {distance}")
                    #     continue

            if mol.spin == 0:
                cc(mol, grids, name, args, evaluate=evaluate)
            else:
                ucc(mol, grids, name, args, evaluate=evaluate)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {name_mol} {extend_atom} {extend_xyz} {distance}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol} {extend_atom} {extend_xyz} {distance}")
        print()

    print(f"Error molecule: {error_molecule}")
