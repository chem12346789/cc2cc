"""Molecular dict"""

import copy
import json
import os
import importlib.resources
from pathlib import Path

import numpy as np

import pyscf.gto

from cc2cc.utils.basis import gen_basis
from cc2cc.utils.rotate import rotate

AU2KCALMOL = 627.5096080306
AU2DEBYE = 2.541746
dataset = {}

with importlib.resources.path("cc2cc", "utils") as resource_path:
    for dataset_name in Path(os.fspath(resource_path)).rglob("*.json"):
        with open(
            Path(os.fspath(resource_path)) / f"{dataset_name.stem}.json",
            "r",
            encoding="utf-8",
        ) as f:
            dataset[dataset_name.stem] = json.load(f)


def extend(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    basis: str,
    dataset_name: str = "Mol",
) -> tuple:
    """
    Function to extend the molecular
    """
    molecular = np.array(copy.deepcopy(dataset[dataset_name][name_mol]), dtype=object)
    print(f"Generate {name_mol}_{distance:.4f}")
    print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}")
    print("original mol", molecular)
    name = f"{name_mol}_{basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

    if "-" in extend_atom:
        atom_list_1 = np.array(extend_atom.split("-")[0].split("."), dtype=int)
        atom_list_2 = np.array(extend_atom.split("-")[1].split("."), dtype=int)

        distance_1_2_array = (
            molecular[atom_list_2[0]][1:4] - molecular[atom_list_1[0]][1:4]
        )
        distance_1_2 = np.linalg.norm(distance_1_2_array)
        molecular[atom_list_2, 1:] = molecular[atom_list_2, 1:] + (
            distance * distance_1_2_array / distance_1_2
        )
    else:
        extend_atom = int(extend_atom)
        molecular[extend_atom][extend_xyz] += distance
    print("extend mol", molecular)
    rotate(molecular, rotation="r", verbose=True)
    return list(molecular), name


def gen_mole(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    basis: str,
    if_basis_str: bool,
    dataset_name: str = "Mol",
) -> pyscf.gto.Mole:
    """
    Function to generate the molecule
    """
    try:
        molecular, name = extend(
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
            basis,
            dataset_name,
        )
    except Exception as e:
        print(f"Error: {name_mol} {extend_atom} {extend_xyz} {distance}")
        print(e)
        return None, None

    mol = pyscf.M(
        atom=molecular,
        basis=gen_basis(
            molecular,
            basis,
            if_basis_str,
        ),
        verbose=4,
        spin=dataset[dataset_name]["spin"][name_mol],
        charge=dataset[dataset_name]["charge"][name_mol],
    )

    return mol, name
