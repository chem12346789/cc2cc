"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product

import torch

from dft2cc import add_args, extend
from dft2cc import test_rks, test_uks

from dft2cc.utils import ModelDict

# from cadft.utils.ModelDict_xy import ModelDict
# from cadft.utils import ModelDict_xy1 as ModelDict
# from cadft.utils.ModelDict_xy2 import ModelDict

# class ModelDict_data()
if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init the model
    modeldict = ModelDict(
        load=args.load,
        device=device,
        precision=args.precision,
        if_mkdir=False,
        load_epoch=args.load_epoch,
    )
    modeldict.load_model()
    modeldict.eval()

    # 2. Test loop
    df_dict = {
        "name": [],
        "error_scf_ene": [],
        "error_dft_ene": [],
        "abs_cc_ene": [],
        "dipole_diff_scf": [],
        "dipole_diff_dft": [],
        "force_diff_scf": [],
        "force_diff_dft": [],
    }

    name_mol_now = args.name_mol[0]

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
        molecular, name = extend(
            name_mol, extend_atom, extend_xyz, distance, args.basis
        )
        if molecular is None:
            print(f"Skip: {name:>40}")
            continue

        if "openshell" in name_mol:
            test_uks(args, molecular, name, modeldict, df_dict)
        else:
            test_rks(args, molecular, name, modeldict, df_dict)
