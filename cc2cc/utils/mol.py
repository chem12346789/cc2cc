"""molecule dict"""

import copy
import json
import os
import importlib.resources
from pathlib import Path
import re

import numpy as np

import basis_set_exchange
import pyscf
from pyscf.gto.basis.bse import get_basis
from pyscf.gto.basis.bse import _ecp_basis

from cc2cc.utils.rotate import rotate
from cc2cc.utils.addon_basis import addon_basis

AU2KCALMOL = 627.5094733748099
AU2DEBYE = 2.541746472
dataset = {}
with importlib.resources.path("cc2cc", "utils") as resource_path:
    for dataset_name_ in Path(os.fspath(resource_path)).rglob("*.json"):
        with open(
            Path(os.fspath(resource_path)) / f"{dataset_name_.stem}.json",
            "r",
            encoding="utf-8",
        ) as f:
            dataset[dataset_name_.stem] = json.load(f)


def get_ecp(name, elements):
    """
    Obtain the effective core potential (ECP) from Basis Set Exchange.

    Args:
        name : str
            Name of the basis set, case insensitive.
        elements : str, int or list

    Returns:
        A dict of ECP data for the specified elements.
    """
    basis = basis_set_exchange.api.get_basis(name, elements)
    return _ecp_basis(basis)


def_ecp_list = [
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
]
aug_cc_atom_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
]
aug_cc_pp_atom_list = def_ecp_list.copy()
aug_cc_pp_atom_list.remove("Rb")
aug_cc_pp_atom_list.remove("Sr")
aug_cc_pp_atom_list.remove("Cs")
aug_cc_pp_atom_list.remove("Ba")
aug_cc_pp_atom_list.remove("La")
aug_cc_atom_list.remove("K")
aug_cc_atom_list.remove("Ca")
special_atom_list = ["K", "Ca"]
cc_atom_list = aug_cc_atom_list.copy()
cc_pp_atom_list = aug_cc_pp_atom_list.copy()


def gen_basis(basis: str, atom_list: list):
    """
    Function to generate the basis
    """
    if "def2" in basis:
        return basis
    if "cc-" in basis:
        dict_ = {}
        for atom in atom_list:
            if atom in dict_:
                continue
            if "aug-" in basis:
                if atom in aug_cc_atom_list:
                    dict_.update(get_basis(basis, atom))
                if atom in aug_cc_pp_atom_list:
                    dict_.update(get_basis(basis + "-PP", atom))
                if atom in special_atom_list:
                    dict_.update(addon_basis[atom][basis])
            else:
                if atom in cc_atom_list:
                    dict_.update(get_basis(basis, atom))
                if atom in cc_pp_atom_list:
                    dict_.update(get_basis(basis + "-PP", atom))
                if atom in special_atom_list:
                    dict_.update(addon_basis[atom][basis])
        return dict_


def gen_ecp(basis: str, atom_list: list) -> dict:
    """
    Function to generate the ecp
    """
    if "def2" in basis:
        dict_ = {}
        for atom in def_ecp_list:
            dict_[atom] = basis
        return dict_
    if "cc-" in basis:
        dict_ = {}
        for atom in atom_list:
            if atom in dict_:
                continue
            if "aug-" in basis:
                if atom in aug_cc_pp_atom_list:
                    dict_.update(get_ecp(basis + "-PP", atom))
            else:
                if atom in cc_pp_atom_list:
                    dict_.update(get_ecp(basis + "-PP", atom))
        return dict_


def extend(
    name_mol: str,
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

    if verbose > 3:
        print("original mol", molecule)

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
    basis: str,
    dataset_name: str,
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
        dataset_name,
        verbose=verbose,
        if_rotate=if_rotate,
        if_rotate_random=if_rotate_random,
        solve_symmetry=solve_symmetry,
    )

    match = re.search(r"\(.*\)", basis)
    if match:
        if dataset[dataset_name]["charge"][name_mol] >= 0:
            basis = basis[: match.start()] + basis[match.end() :]
        else:
            basis = basis.replace("(", "").replace(")", "")

    mol = pyscf.M(
        atom=molecule,
        basis=gen_basis(basis, [atom[0] for atom in molecule]),
        ecp=gen_ecp(basis, [atom[0] for atom in molecule]),
        verbose=verbose,
        spin=dataset[dataset_name]["spin"][name_mol],
        charge=dataset[dataset_name]["charge"][name_mol],
    )

    return mol
