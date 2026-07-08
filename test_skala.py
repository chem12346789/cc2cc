"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from pathlib import Path
import json
import copy
import re

import pyscf
import numpy as np
import gpu4pyscf

# from skala.pyscf import SkalaKS
from skala.gpu4pyscf import SkalaKS

dataset = {}
json_dir = Path(__file__).resolve().parent / "cc2cc/utils/mol_dataset"
print(f"Loading dataset from {json_dir}")
for dataset_name_ in json_dir.glob("*.json"):
    with dataset_name_.open("r", encoding="utf-8") as f:
        dataset[dataset_name_.stem] = json.load(f)
print(f"Loaded datasets: {list(dataset.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the model or benchmark DFT calculations. Other parameters are from the argparse."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mol",
        help="Name of the dataset. Default is mol (training and testing).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="cc-pVDZ",
        help="Basis set for the calculation.",
    )
    parser.add_argument(
        "--name_mol",
        "-m",
        nargs="+",
        type=str,
        default=[],
        help="Name of molecule. Default is None (all the dataset).",
    )
    args = parser.parse_args()

    for name_mol in args.name_mol:
        print(f"Testing molecule: {name_mol}")
        mol = None
        try:
            molecule = np.array(
                copy.deepcopy(dataset[args.dataset][name_mol]), dtype=object
            )

            basis = args.basis
            match = re.search(r"\(.*\)", basis)
            if match:
                if dataset[args.dataset]["charge"][name_mol] >= 0:
                    basis = basis[: match.start()] + basis[match.end() :]
                else:
                    basis = basis.replace("(", "").replace(")", "")

            print(f"Using basis set: {basis}")
            print(f"and molecule: {molecule}")
            mol = pyscf.M(
                atom=list(molecule),
                basis=basis,
                ecp=basis,
                spin=dataset[args.dataset]["spin"][name_mol],
                charge=dataset[args.dataset]["charge"][name_mol],
            )

            mf = SkalaKS(mol, xc="skala-1.1")
            # mf = pyscf.dft.RKS(mol).to_gpu().density_fit()
            mf.conv_tol = 1e-7
            mf.conv_tol_grad = 1e-3
            mf.verbose = 9
            mf.kernel()

        finally:
            del mol


if __name__ == "__main__":
    main()
