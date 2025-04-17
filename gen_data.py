""" """

import argparse
from itertools import product

from cc2cc import add_args, cc, ucc
from cc2cc.utils import Grid, gen_mole, print_gpu_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    error_molecule = []

    print_gpu_info(args.device)

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

            if mol.spin == 0:
                cc(mol, grids, name)
            else:
                ucc(mol, grids, name)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {name_mol} {extend_atom} {extend_xyz} {distance}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol} {extend_atom} {extend_xyz} {distance}")
        print()

    print(f"Error molecule: {error_molecule}")
