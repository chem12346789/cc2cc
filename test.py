"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product

import torch

from cc2cc import add_args, test_rks, test_uks
from cc2cc.utils import gen_mole

from cc2cc.utils import Model_Dict, Data_Record

from cc2cc.utils import MAIN_PATH


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
    modeldict = Model_Dict(
        load=args.load,
        device=device,
        precision=args.precision,
        if_mkdir=False,
        load_epoch=args.load_epoch,
    )
    modeldict.load_model()
    modeldict.eval()

    # 2. Test loop
    data_record = Data_Record(MAIN_PATH / f"validate/ccdft_{args.load}.csv")

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
        mol, name = gen_mole(
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
            args.basis,
            args.if_basis_str,
        )

        if mol.spin == 0:
            test_rks(mol, name, modeldict, data_record)
        else:
            test_uks(mol, name, modeldict, data_record)
