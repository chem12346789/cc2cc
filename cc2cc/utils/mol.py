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


def2_ecp_basis = {
    "Rb": "def2-ECP",
    "Sr": "def2-ECP",
    "Y": "def2-ECP",
    "Zr": "def2-ECP",
    "Nb": "def2-ECP",
    "Mo": "def2-ECP",
    "Tc": "def2-ECP",
    "Ru": "def2-ECP",
    "Rh": "def2-ECP",
    "Pd": "def2-ECP",
    "Ag": "def2-ECP",
    "Cd": "def2-ECP",
    "In": "def2-ECP",
    "Sn": "def2-ECP",
    "Sb": "def2-ECP",
    "Te": "def2-ECP",
    "I": "def2-ECP",
    "Xe": "def2-ECP",
    "Cs": "def2-ECP",
    "Ba": "def2-ECP",
    "La": "def2-ECP",
    "Hf": "def2-ECP",
    "Ta": "def2-ECP",
    "W": "def2-ECP",
    "Re": "def2-ECP",
    "Os": "def2-ECP",
    "Ir": "def2-ECP",
    "Pt": "def2-ECP",
    "Au": "def2-ECP",
    "Hg": "def2-ECP",
    "Tl": "def2-ECP",
    "Pb": "def2-ECP",
    "Bi": "def2-ECP",
    "Po": "def2-ECP",
    "At": "def2-ECP",
    "Rn": "def2-ECP",
}

ma_basis = {
    "def2-svpd": "madef2svp",
    "def2-tzvppd": "madef2tzvpp",
    "def2-tzvpd": "madef2tzvp",
    "def2-qzvppd": "madef2qzvpp",
    "def2-qzvpd": "madef2qzvp",
}


def gen_basis(basis):
    return {
        "H": basis,
        "He": basis,
        "Li": basis,
        "Be": basis,
        "B": basis,
        "C": basis,
        "N": basis,
        "O": basis,
        "F": basis,
        "Ne": basis,
        "Na": basis,
        "Mg": basis,
        "Al": basis,
        "Si": basis,
        "P": basis,
        "S": basis,
        "Cl": basis,
        "Ar": basis,
        # "K": basis,
        "Ca": basis,
        "Sc": basis,
        "Ti": basis,
        "V": basis,
        "Cr": basis,
        "Mn": basis,
        "Fe": basis,
        "Co": basis,
        "Ni": basis,
        "Cu": basis,
        "Zn": basis,
        "Ga": basis,
        "Ge": basis,
        "As": basis,
        "Se": basis,
        "Br": basis,
        "Kr": basis,
        "Y": f"{basis}-PP",
        "Zr": f"{basis}-PP",
        "Nb": f"{basis}-PP",
        "Mo": f"{basis}-PP",
        "Tc": f"{basis}-PP",
        "Ru": f"{basis}-PP",
        "Rh": f"{basis}-PP",
        "Pd": f"{basis}-PP",
        "Ag": f"{basis}-PP",
        "Cd": f"{basis}-PP",
        "In": f"{basis}-PP",
        "Sn": f"{basis}-PP",
        "Sb": f"{basis}-PP",
        "Te": f"{basis}-PP",
        "I": f"{basis}-PP",
        "Xe": f"{basis}-PP",
        "Hf": f"{basis}-PP",
        "Ta": f"{basis}-PP",
        "W": f"{basis}-PP",
        "Re": f"{basis}-PP",
        "Os": f"{basis}-PP",
        "Ir": f"{basis}-PP",
        "Pt": f"{basis}-PP",
        "Au": f"{basis}-PP",
        "Hg": f"{basis}-PP",
        "Tl": f"{basis}-PP",
        "Pb": f"{basis}-PP",
        "Bi": f"{basis}-PP",
        "Po": f"{basis}-PP",
        "At": f"{basis}-PP",
        "Rn": f"{basis}-PP",
    }


def gen_ecp(basis):
    return {
        "Y": f"{basis}-PP",
        "Zr": f"{basis}-PP",
        "Nb": f"{basis}-PP",
        "Mo": f"{basis}-PP",
        "Tc": f"{basis}-PP",
        "Ru": f"{basis}-PP",
        "Rh": f"{basis}-PP",
        "Pd": f"{basis}-PP",
        "Ag": f"{basis}-PP",
        "Cd": f"{basis}-PP",
        "In": f"{basis}-PP",
        "Sn": f"{basis}-PP",
        "Sb": f"{basis}-PP",
        "Te": f"{basis}-PP",
        "I": f"{basis}-PP",
        "Xe": f"{basis}-PP",
        "Hf": f"{basis}-PP",
        "Ta": f"{basis}-PP",
        "W": f"{basis}-PP",
        "Re": f"{basis}-PP",
        "Os": f"{basis}-PP",
        "Ir": f"{basis}-PP",
        "Pt": f"{basis}-PP",
        "Au": f"{basis}-PP",
        "Hg": f"{basis}-PP",
        "Tl": f"{basis}-PP",
        "Pb": f"{basis}-PP",
        "Bi": f"{basis}-PP",
        "Po": f"{basis}-PP",
        "At": f"{basis}-PP",
        "Rn": f"{basis}-PP",
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
        if abs(distance) > 1e-12:
            raise ValueError("Distance is not allowed in single atom")
    else:
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
    if verbose > 3:
        print(
            f"if_rotate:{if_rotate} if_rotate_random:{if_rotate_random} solve_symmetry:{solve_symmetry}"
        )
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
        basis=gen_basis(basis) if "cc" in dataset_name else basis,
        ecp=gen_ecp(basis) if "cc" in dataset_name else def2_ecp_basis,
        verbose=verbose,
        spin=dataset[dataset_name]["spin"][name_mol],
        charge=dataset[dataset_name]["charge"][name_mol],
    )

    return mol
