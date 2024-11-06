"""

"""

import argparse
from itertools import product

from cc2cc import add_args, extend, cc, cc_change_cube, cc_add_data
from cc2cc.utils import rotate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

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
        molecular, name = extend(
            name_mol, extend_atom, extend_xyz, distance, args.basis
        )
        rotate(molecular)

        if "open-shell" in name:
            continue
        else:
            cc(molecular, name, args)
            # cc_add_data(molecular, name, args)
            # cc_change_cube(molecular, name, args)
