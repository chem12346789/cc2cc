"""molecule dict"""

import copy
import json
from pathlib import Path
import re

import numpy as np

import pyscf

from cc2cc.utils.helper.chemistry.basis import (
    aug_cc_atom_list,
    aug_cc_pp_atom_list,
    cc_atom_list,
    cc_pp_atom_list,
    def_ecp_list,
    gen_basis,
    gen_ecp,
    gmtkn55_qzvp_basis,
    get_ecp,
    special_atom_list,
)
from cc2cc.utils.helper.chemistry.rotate import rotate
from cc2cc.utils.helper.units import AU2DEBYE, AU2KCALMOL

dataset = {}
json_dir = Path(__file__).resolve().parent / "mol_dataset"
for dataset_name_ in json_dir.glob("*.json"):
    with dataset_name_.open("r", encoding="utf-8") as f:
        dataset[dataset_name_.stem] = json.load(f)


def rotate_mol(
    name_mol: str,
    dataset_name: str = "Mol",
    verbose=4,
    if_rotate=False,
    if_rotate_random=False,
    solve_symmetry=False,
) -> tuple:
    """
    Function to rotate the molecule
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
    molecule = rotate_mol(
        name_mol,
        dataset_name,
        verbose=verbose,
        if_rotate=if_rotate,
        if_rotate_random=if_rotate_random,
        solve_symmetry=solve_symmetry,
    )

    atom_list = [atom[0] for atom in molecule]
    if basis == "def2-QZVP-GMTKN55":
        if dataset_name not in ("gmtkn-def2", "gmtkn-diet100-def2", "gmtkn-diet30-def2"):
            raise ValueError(f"{basis} requires a GMTKN dataset, got {dataset_name}.")
        mole_basis = gmtkn55_qzvp_basis(atom_list, name_mol.split("-", 1)[0])
        ecp = gen_ecp("def2-QZVP", atom_list)
    else:
        match = re.search(r"\(.*\)", basis)
        if match:
            if dataset[dataset_name]["charge"][name_mol] >= 0:
                basis = basis[: match.start()] + basis[match.end() :]
            else:
                basis = basis.replace("(", "").replace(")", "")
        mole_basis = gen_basis(basis, atom_list)
        ecp = gen_ecp(basis, atom_list)

    mol = pyscf.M(
        atom=molecule,
        basis=mole_basis,
        ecp=ecp,
        verbose=verbose,
        spin=dataset[dataset_name]["spin"][name_mol],
        charge=dataset[dataset_name]["charge"][name_mol],
    )

    return mol
