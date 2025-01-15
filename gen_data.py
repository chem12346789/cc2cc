"""

"""

import argparse
from itertools import product
import os

from cc2cc import add_args, cc, ucc
from cc2cc.utils import gen_mole


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

        if mol.spin == 0:
            cc(mol, name)
        else:
            ucc(mol, name)

        print()
