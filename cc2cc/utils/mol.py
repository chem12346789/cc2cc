"""Molecular dict"""

import copy
import importlib.resources
import json
from pathlib import Path
import os

import pyscf.gto

from cc2cc.utils.basis import gen_basis

AU2KCALMOL = 627.5096080306
AU2DEBYE = 2.541746
dataset = {}

with importlib.resources.path("cc2cc", "utils") as resource_path:
    for dataset_name in Path(os.fspath(resource_path)).rglob("mol.json")
    with open(
        Path(os.fspath(resource_path)) / "mol.json",
        "r",
        encoding="utf-8",
    ) as f:
        dataset.update(json.load(f))


def extend(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    basis: str,
) -> tuple:
    """
    Function to extend the molecular
    """
    molecular = copy.deepcopy(Mol[name_mol])
    print(f"Generate {name_mol}_{distance:.4f}")
    print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}")
    print("original mol", molecular)
    name = f"{name_mol}_{basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

    if "-" in extend_atom:
        if "." in extend_atom:
            extend_atom_1_l = [
                int(i_atom) for i_atom in extend_atom.split("_")[0].split(".")
            ]
            extend_atom_2_l = [
                int(i_atom) for i_atom in extend_atom.split("_")[1].split(".")
            ]
            print(extend_atom_1_l, extend_atom_2_l)
            for extend_i in extend_atom_1_l:
                if extend_i >= len(Mol[name_mol]):
                    print(f"Skip: {name:>40}")
                    return None, name
            for extend_i in extend_atom_2_l:
                if extend_i >= len(Mol[name_mol]):
                    print(f"Skip: {name:>40}")
                    return None, name
            if abs(distance) < 1e-3:
                return None, name
            distance_1_2_array = [
                molecular[extend_atom_2_l[0]][1] - molecular[extend_atom_1_l[0]][1],
                molecular[extend_atom_2_l[0]][2] - molecular[extend_atom_1_l[0]][2],
                molecular[extend_atom_2_l[0]][3] - molecular[extend_atom_1_l[0]][3],
            ]
            distance_1_2 = sum(map(lambda x: x**2, distance_1_2_array)) ** 0.5
            for i in range(1, 4):
                for extend_i in extend_atom_2_l:
                    molecular[extend_i][i] += (
                        distance * distance_1_2_array[i - 1] / distance_1_2
                    )
        else:
            extend_atom_1, extend_atom_2 = map(int, extend_atom.split("-"))
            if extend_atom_1 >= len(Mol[name_mol]) or extend_atom_2 >= len(
                Mol[name_mol]
            ):
                print(f"Skip: {name:>40}")
                return None, name
            if abs(distance) < 1e-3:
                if (extend_atom_1 != 0) and (extend_atom_2 != 1):
                    print(f"Skip: {name:>40}")
                    return None, name
            distance_1_2_array = [
                molecular[extend_atom_2][1] - molecular[extend_atom_1][1],
                molecular[extend_atom_2][2] - molecular[extend_atom_1][2],
                molecular[extend_atom_2][3] - molecular[extend_atom_1][3],
            ]
            distance_1_2 = sum(map(lambda x: x**2, distance_1_2_array)) ** 0.5
            for i in range(1, 4):
                molecular[extend_atom_2][i] += (
                    distance * distance_1_2_array[i - 1] / distance_1_2
                )
    else:
        extend_atom = int(extend_atom)
        if abs(distance) < 1e-3:
            if (extend_atom != 0) or extend_xyz != 1:
                print(f"Skip: {name:>40}")
                return None, name
        if extend_atom >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            return None, name
        molecular[extend_atom][extend_xyz] += distance
    print("extend mol", molecular)
    return molecular, name


def gen_mole(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    basis: str,
    if_basis_str: bool,
) -> pyscf.gto.Mole:
    """
    Function to generate the molecule
    """
    molecular, name = extend(name_mol, extend_atom, extend_xyz, distance, basis)

    mol = pyscf.M(
        atom=molecular,
        basis=gen_basis(
            molecular,
            basis,
            if_basis_str,
        ),
        verbose=3,
        spin=Mol["spin"][name_mol],
        charge=Mol["charge"][name_mol],
    )

    return mol, name
