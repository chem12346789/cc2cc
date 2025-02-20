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
from cc2cc.utils import add_args
from cc2cc.utils import Grid, ModelDict, DataRecord, DataBase
from cc2cc.utils import MAIN_PATH, AU2KCALMOL

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

    total_str_list = train_str_list + eval_str_list

    database_total = DataBase(total_str_list, args)

    loss_ene_l, loss_ene_abs_l = [], []

    for name in database_total.name_list:
        number_batch_name = 0
        loss_ene_name = 0.0
        loss_ene_abs_name = 0.0

        for batch in database_total.data_gpu[name]:
            with torch.no_grad():
                input_mat = batch["input"]
                weight = batch["weight"]
                output_mat_real = batch["output"]
                output_mat = modeldict.model(input_mat)
                loss_ene_mat = output_mat_real * weight - output_mat * weight
                loss_ene = torch.sum(loss_ene_mat)
                loss_ene_abs = torch.sum(torch.abs(loss_ene_mat))
            number_batch_name += len(batch["weight"])
            loss_ene_name += loss_ene.item()
            loss_ene_abs_name += loss_ene_abs.item() * len(batch["weight"])
            print(
                f"Name: {name}, loss_ene: {AU2KCALMOL * loss_ene.item()}, loss_ene_abs: {AU2KCALMOL * loss_ene_abs.item()}"
            )

        loss_ene_l.append(AU2KCALMOL * loss_ene_name / number_batch_name)
        loss_ene_abs_l.append(AU2KCALMOL * np.abs(loss_ene_abs_name))

        data_record.add_data(
            name,
            {
                "loss_ene_l": loss_ene_l[-1],
                "loss_ene_abs_l": loss_ene_abs_l[-1],
            },
        )
        data_record.save_csv()
