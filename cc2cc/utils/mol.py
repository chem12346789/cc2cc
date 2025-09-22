"""molecule dict"""

import copy
import json
import os
import importlib.resources
from pathlib import Path

import numpy as np

import pyscf

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
        "K": pyscf.gto.basis.parse(
            """
 K s
 2.709796E+07  4.136110E-07 -1.188290E-07  3.917470E-08 -7.564440E-09  0.000000E+00  0.000000E+00
 3.592856E+06  3.860330E-06 -1.108550E-06  3.650640E-07 -7.055390E-08  0.000000E+00  0.000000E+00
 7.222638E+05  2.394640E-05 -6.881230E-06  2.269460E-06 -4.380820E-07  0.000000E+00  0.000000E+00
 1.833976E+05  1.157540E-04 -3.324150E-05  1.094290E-05 -2.115540E-06  0.000000E+00  0.000000E+00
 5.464812E+04  4.697010E-04 -1.350850E-04  4.457360E-05 -8.601300E-06  0.000000E+00  0.000000E+00
 1.830913E+04  1.673720E-03 -4.814570E-04  1.584480E-04 -3.064440E-05  0.000000E+00  0.000000E+00
 6.712501E+03  5.370450E-03 -1.552620E-03  5.128440E-04 -9.892550E-05  0.000000E+00  0.000000E+00
 2.643719E+03  1.572510E-02 -4.578330E-03  1.507690E-03 -2.917760E-04  0.000000E+00  0.000000E+00
 1.103835E+03  4.193360E-02 -1.247930E-02  4.136940E-03 -7.977860E-04  0.000000E+00  0.000000E+00
 4.836777E+02  9.987620E-02 -3.094650E-02  1.024690E-02 -1.985820E-03  0.000000E+00  0.000000E+00
 2.206245E+02  2.019680E-01 -6.855530E-02  2.306520E-02 -4.449740E-03  0.000000E+00  0.000000E+00
 1.041117E+02  3.146270E-01 -1.268300E-01  4.308900E-02 -8.390260E-03  0.000000E+00  0.000000E+00
 5.052749E+01  3.107540E-01 -1.745020E-01  6.238120E-02 -1.206000E-02  0.000000E+00  0.000000E+00
 2.485247E+01  1.421920E-01 -9.089560E-02  3.270420E-02 -6.568410E-03  0.000000E+00  0.000000E+00
 1.128102E+01  1.658490E-02  2.506960E-01 -1.021980E-01  2.061430E-02  0.000000E+00  0.000000E+00
 5.510166E+00 -2.759760E-04  5.589630E-01 -3.317970E-01  6.770060E-02  0.000000E+00  0.000000E+00
 2.706914E+00  5.158310E-04  3.284370E-01 -3.030100E-01  6.639570E-02  0.000000E+00  0.000000E+00
 1.122807E+00 -2.268950E-04  3.392170E-02  3.098410E-01 -8.237770E-02  0.000000E+00  0.000000E+00
 5.236385E-01  6.652980E-05 -2.100280E-03  6.732700E-01 -1.727150E-01  0.000000E+00  0.000000E+00
 2.355180E-01 -3.685150E-05  1.383960E-03  2.812200E-01 -2.310290E-01  0.000000E+00  0.000000E+00
 4.157977E-02  1.925250E-05 -5.315630E-04  9.376240E-03  5.090040E-01  0.000000E+00  0.000000E+00
 2.555456E-02 -2.108420E-05  5.770310E-04 -6.955720E-03  3.472180E-01  1.000000E+00  0.000000E+00
 1.471495E-02  7.044600E-06 -1.931440E-04  2.069460E-03  2.809620E-01  0.000000E+00  1.000000E+00
 K p
 9.533309E+03  4.212010E-05 -1.280760E-05  1.764010E-06  0.000000E+00  0.000000E+00
 1.852081E+03  5.068300E-04 -1.547870E-04  2.124880E-05  0.000000E+00  0.000000E+00
 5.464251E+02  3.311700E-03 -1.010940E-03  1.392500E-04  0.000000E+00  0.000000E+00
 1.966947E+02  1.536840E-02 -4.738330E-03  6.509140E-04  0.000000E+00  0.000000E+00
 7.964112E+01  5.457060E-02 -1.703260E-02  2.349940E-03  0.000000E+00  0.000000E+00
 3.488324E+01  1.473880E-01 -4.754130E-02  6.547470E-03  0.000000E+00  0.000000E+00
 1.604479E+01  2.879240E-01 -9.560810E-02  1.326270E-02  0.000000E+00  0.000000E+00
 7.575632E+00  3.750810E-01 -1.337000E-01  1.848520E-02  0.000000E+00  0.000000E+00
 3.672548E+00  2.545160E-01 -7.425060E-02  1.025530E-02  0.000000E+00  0.000000E+00
 1.771130E+00  6.059280E-02  1.646810E-01 -2.652150E-02  0.000000E+00  0.000000E+00
 8.515199E-01  2.508370E-03  4.057630E-01 -6.086460E-02  0.000000E+00  0.000000E+00
 4.016017E-01  6.688870E-04  4.113600E-01 -7.474220E-02  0.000000E+00  0.000000E+00
 1.851933E-01 -1.486150E-04  1.734240E-01 -4.870880E-02  0.000000E+00  0.000000E+00
 5.523390E-02  4.936010E-05  1.253640E-02  2.245720E-01  1.000000E+00  0.000000E+00
 2.431085E-02 -3.513850E-05 -3.267140E-03  5.839280E-01  0.000000E+00  0.000000E+00
 1.080339E-02  1.171450E-05  1.039680E-03  2.980960E-01  0.000000E+00  1.000000E+00
 K d
 5.670903E+00  8.285900E-03  0.000000E+00  0.000000E+00
 1.459042E+00  2.332530E-02  0.000000E+00  0.000000E+00
 5.431827E-01  4.878080E-02  0.000000E+00  0.000000E+00
 1.346875E-01  7.349020E-02  0.000000E+00  0.000000E+00
 5.516013E-02  2.387630E-01  1.000000E+00  0.000000E+00
 1.501876E-02  8.253200E-01  0.000000E+00  1.000000E+00
 K f
 8.493750E-02  1.000000E+00
"""
        ),
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
        basis=gen_basis(basis) if "cc" in basis else basis,
        ecp=gen_ecp(basis) if "cc" in basis else def2_ecp_basis,
        verbose=verbose,
        spin=dataset[dataset_name]["spin"][name_mol],
        charge=dataset[dataset_name]["charge"][name_mol],
    )

    return mol
