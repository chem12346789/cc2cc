""" """

import argparse
from itertools import product

from cc2cc import add_args, cc, ucc
from cc2cc.utils import Grid, gen_mole, print_computer_info
from cc2cc.utils.env_var import DATA_PATH
from cc2cc.utils.parser import gen_name_args

train_str_list = [
    "molecule0",
    "molecule1",
    "molecule2",
    "molecule3-ALK8",
    "molecule3-HEAVYSB11",
    "molecule3-W4_11",
    "molecule3-AL2X6",
    "molecule4-ALK8",
    "molecule4-W4_11",
    "BH9-08_9R2",  # 5
]
train_str_exclude_list = [
    "W4_11-propane",  # 3
    "molecule1-ACC24",
    "molecule1-GAPS",
    "molecule1-GW100",
    "molecule1-MRADC",
    "molecule1-S30L",
    "molecule2-ACC24",
    "molecule2-GAPS",
    "molecule2-GW100",
    "molecule2-MRADC",
    "molecule2-S30L",
    "molecule3-ACC24",
    "molecule3-GAPS",
    "molecule3-GW100",
    "molecule3-MRADC",
    "molecule3-S30L",
    "molecule4-ACC24",
    "molecule4-GAPS",
    "molecule4-GW100",
    "molecule4-MRADC",
    "molecule4-S30L",
]
eval_str_list = [
    "ADIM6-AD2",  # 4
    "molecule5-ALK8",
    "molecule5-BSR36",
    "molecule5-W4_11",
    "molecule6-BSR36",
    "ADIM6-AD3",  # 6
    # "molecule7-MB16_43",
    # "molecule7-BSR36",
    # "molecule8-MB16_43",
    # "ADIM6-AD4",  # 8
    # "molecule8-BSR36",
    # "molecule8-IDISP",
]
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

    # name_mol_list = train_str_list
    name_mol_list = eval_str_list
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
                args.if_basis_str,
                args.dataset,
            )

            if mol is None:
                print(f"SKIP: {name_mol} {extend_atom} {extend_xyz} {distance}")
                continue

            if args.n_rad is not None and args.n_ang is not None:
                name = f"{name}_{args.n_rad}_{args.n_ang}"
            else:
                name = f"{name}_default"

            grids = Grid(mol, n_rad=args.n_rad, n_ang=args.n_ang)

            if args.if_continue:
                if (DATA_PATH / f"data_{name}.npz").exists():
                    print(f"SKIP: {name_mol} {extend_atom} {extend_xyz} {distance}")
                    continue

            if mol.spin == 0:
                cc(
                    mol,
                    grids,
                    name,
                    args,
                )
            else:
                ucc(
                    mol,
                    grids,
                    name,
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
