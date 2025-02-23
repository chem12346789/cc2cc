"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product
import os

import numpy as np
import torch

from cc2cc import add_args
from cc2cc.utils import ModelDict, DataRecord, DataBase
from cc2cc.utils import MAIN_PATH, AU2KCALMOL
from cc2cc.utils.mol import dataset

from train import TRAIN_STR_LIST as train_str_list, EVAL_STR_LIST as eval_str_list

# from cadft.utils.ModelDict_xy import ModelDict
# from cadft.utils import ModelDict_xy1 as ModelDict
# from cadft.utils.ModelDict_xy2 import ModelDict

if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    print(f"PID: {os.getpid()}")

    # 1. Init the model
    modeldict = ModelDict(args)
    modeldict.load_model()
    modeldict.eval()

    # 2. Test loop
    data_record = DataRecord(
        MAIN_PATH / f"validate/ccdft_{args.basis}_{args.load}_model.csv"
    )

    # total_str_list = train_str_list + eval_str_list
    total_str_list = dataset[args.dataset]["molecular"]
    database_total = DataBase(total_str_list, args)

    error_scf_ene, error_scf_ene_abs = [], []
    error_dft_ene, error_dft_ene_abs = [], []

    for name in database_total.name_list:
        loss_ene_name = 0.0
        loss_ene_abs_name = 0.0
        loss_ene_real_name = 0.0
        loss_ene_real_abs_name = 0.0

        for batch in database_total.data_gpu[name]:
            with torch.no_grad():
                input_mat = batch["input"]
                weight = batch["weight"]
                output_mat_real = batch["output"]
                output_mat = modeldict.model(input_mat)
                loss_ene_mat = output_mat_real * weight - output_mat * weight
                loss_ene_real_mat = output_mat_real * weight
            loss_ene_name += torch.sum(loss_ene_mat).item()
            loss_ene_abs_name += torch.sum(torch.abs(loss_ene_mat)).item()
            loss_ene_real_name += torch.sum(loss_ene_real_mat).item()
            loss_ene_real_abs_name += torch.sum(torch.abs(loss_ene_real_mat)).item()
            print(
                f"Name: {name}, loss_ene: {AU2KCALMOL * loss_ene_name}, loss_ene_abs: {AU2KCALMOL * loss_ene_abs_name}, "
                f"loss_ene_real: {AU2KCALMOL * loss_ene_real_name}, loss_ene_real_abs: {AU2KCALMOL * loss_ene_real_abs_name}"
            )

        error_scf_ene.append(AU2KCALMOL * loss_ene_name)
        error_scf_ene_abs.append(AU2KCALMOL * loss_ene_abs_name)
        error_dft_ene.append(AU2KCALMOL * loss_ene_real_name)
        error_dft_ene_abs.append(AU2KCALMOL * loss_ene_real_abs_name)

        data_record.add_data(
            name,
            {
                "error_scf_ene": error_scf_ene[-1],
                "error_scf_ene_abs": error_scf_ene_abs[-1],
                "error_dft_ene": error_dft_ene[-1],
                "error_dft_ene_abs": error_dft_ene_abs[-1],
                "error_dft_ele": 0.0,
                "error_scf_ele": 0.0,
                "error_dft_dip": 0.0,
                "error_scf_dip": 0.0,
            },
        )
        data_record.save_csv()

    print("Testing process completed. Results saved.")
