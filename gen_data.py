"""

"""

import argparse
from itertools import product

import pyscf

from cc2cc import add_args, extend, cc
from cc2cc.utils import gen_basis, rotate


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
        SPIN = 0
        if "-openshell" in name_mol:
            if "_" in name_mol:
                SPIN = int(name_mol.split("_")[-1])
                name_mol = name_mol.split("_")[0]
                name_mol = name_mol.replace("-openshell", "")
            else:
                SPIN = 1

        molecular, name = extend(
            name_mol, extend_atom, extend_xyz, distance, args.basis
        )
        rotate(molecular, verbose=True)

        mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                args.basis,
                args.if_basis_str,
            ),
            verbose=4,
            spin=0,
        )

        if SPIN == 0:
            cc(mol, name)
        else:
            continue
