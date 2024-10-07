"""Molecular dict"""

import importlib.resources
from pathlib import Path
import os
import json


AU2KCALMOL = 627.5096080306
AU2DEBYE = 2.541746


H_2 = [["H", 0.737, 0, 0], ["H", 0, 0, 0]]
HNHH = [
    ["H", 1.008, -0.0, 0.0],
    ["N", 0.0, 0.0, 0.0],
    ["H", -0.2906, 0.9122, 0.3151],
    ["H", -0.2903, -0.0808, -0.962],
]
HOH = [
    ["H", 1.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["H", -0.2438, 0.9698, 0.0],
]
HOOH = [
    ["H", 0.9688, -0.0, 0.0],
    ["O", 0.0, 0.0, 0.0],
    ["O", -0.241, 0.4457, 1.3617],
    ["H", -0.6906, 1.2843, 1.179],
]

Mol = {
    "H_2": H_2,
    "HH": H_2,
    "He": [["He", 0, 0, 0]],
    "Li": [["Li", 0, 0, 0]],
    "Be": [["Be", 0, 0, 0]],
    "B": [["B", 0, 0, 0]],
    "C": [["C", 0, 0, 0]],
    "O": [["O", 0, 0, 0]],
    "Ne": [["Ne", 0, 0, 0]],
    "N": [["N", 0, 0, 0]],
    "F": [["F", 0, 0, 0]],
    "oxygen": [
        ["O", -1, 0, 0],
        ["O", 1, 0, 0],
    ],
    "methyl": [
        ["C", 0, 0, 0],
        ["H", -1, 0, 0],
        ["H", 0.5, -0.866, 0],
        ["H", 0.5, 0.866, 0],
    ],
    "HOH": HOH,
    "HOOH": HOOH,
    "HNHH": HNHH,
    "HNH2": HNHH,
}

with importlib.resources.path("cc2cc", "utils") as resource_path:
    with open(
        Path(os.fspath(resource_path)) / "mol.json",
        "r",
        encoding="utf-8",
    ) as f:
        Mol.update(json.load(f))

name_mol = []

for name in Mol:
    name_mol.append(name)
