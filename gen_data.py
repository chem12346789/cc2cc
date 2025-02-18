"""

"""

import argparse
from itertools import product
import os

from cc2cc import add_args, cc, ucc
from cc2cc.utils import Grid, gen_mole


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    print(os.getpid())

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
        mol, name = gen_mole(
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

        print()
