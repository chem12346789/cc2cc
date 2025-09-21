"""molecule dict"""

import copy
import json
import os
import importlib.resources
from pathlib import Path

import numpy as np

import pyscf.gto

from cc2cc.utils.rotate import rotate

AU2KCALMOL = 627.5094733748099
AU2DEBYE = 2.541746472
dataset = {}
with importlib.resources.path("cc2cc", "utils") as resource_path:
    for dataset_name in Path(os.fspath(resource_path)).rglob("*.json"):
        with open(
            Path(os.fspath(resource_path)) / f"{dataset_name.stem}.json",
            "r",
            encoding="utf-8",
        ) as f:
            dataset[dataset_name.stem] = json.load(f)


def return_def2_ecp(basis: str) -> dict:
    return {
        "Rb": basis,
        "Sr": basis,
        "Y": basis,
        "Zr": basis,
        "Nb": basis,
        "Mo": basis,
        "Tc": basis,
        "Ru": basis,
        "Rh": basis,
        "Pd": basis,
        "Ag": basis,
        "Cd": basis,
        "In": basis,
        "Sn": basis,
        "Sb": basis,
        "Te": basis,
        "I": basis,
        "Xe": basis,
        "Cs": basis,
        "Ba": basis,
        "La": basis,
        "Hf": basis,
        "Ta": basis,
        "W": basis,
        "Re": basis,
        "Os": basis,
        "Ir": basis,
        "Pt": basis,
        "Au": basis,
        "Hg": basis,
        "Tl": basis,
        "Pb": basis,
        "Bi": basis,
        "Po": basis,
        "At": basis,
        "Rn": basis,
    }


def extend(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    dataset_name: str = "Mol",
    verbose=4,
    if_rotate=False,
    if_rotate_random=False,
    solve_symmetry=False,
) -> tuple:
    """
    Function to extend the molecule
    """
    molecule = np.array(copy.deepcopy(dataset[dataset_name][name_mol]), dtype=object)

    if isinstance(extend_atom, int):
        extend_atom = str(extend_atom)

    if verbose > 3:
        print(f"Generate {name_mol}_{distance:.4f}")
        print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}")
        print("original mol", molecule)

    if len(molecule) == 1:
        if abs(distance) < 1e-5:
            return list(molecule)
        else:
            raise ValueError("Distance is not allowed in single atom")

    if "-" in extend_atom:
        atom_list_1 = np.array(extend_atom.split("-")[0].split("."), dtype=int)
        atom_list_2 = np.array(extend_atom.split("-")[1].split("."), dtype=int)

        distance_1_2_array = (
            molecule[atom_list_2[0]][1:4] - molecule[atom_list_1[0]][1:4]
        )
        molecule[atom_list_2, 1:] += distance * distance_1_2_array
    else:
        extend_atom = int(extend_atom)
        molecule[extend_atom][extend_xyz] += distance
    if verbose > 3:
        print("extend mol", molecule)
    if if_rotate:
        if if_rotate_random:
            molecule, _ = rotate(
                molecule,
                test_rotation="random",
                solve_symmetry=solve_symmetry,
                verbose=verbose,
            )
        else:
            molecule, _ = rotate(
                molecule,
                solve_symmetry=solve_symmetry,
                verbose=verbose,
            )
    return list(molecule)


def gen_mole(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    basis: str,
    dataset_name: str = "Mol",
    verbose=0,
    if_rotate=False,
    if_rotate_random=False,
    solve_symmetry=False,
) -> pyscf.gto.Mole:
    """
    Function to generate the molecule
    """
    molecule = extend(
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
        dataset_name,
        verbose=verbose,
        if_rotate=if_rotate,
        if_rotate_random=if_rotate_random,
        solve_symmetry=solve_symmetry,
    )

    mol = pyscf.M(
        atom=molecule,
        basis=basis,
        ecp=return_def2_ecp(basis),
        verbose=verbose,
        spin=dataset[dataset_name]["spin"][name_mol],
        charge=dataset[dataset_name]["charge"][name_mol],
    )

    return mol
