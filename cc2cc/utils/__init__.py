import pandas as pd

from cc2cc.utils.basis import gen_basis
from cc2cc.utils.parser import add_args
from cc2cc.utils.mol import Mol, AU2KCALMOL, AU2DEBYE
from cc2cc.utils.mol import extend
from cc2cc.utils.env_var import (
    MAIN_PATH,
    DATA_PATH,
    DATA_SCF_PATH,
    DATA_TEST_PATH,
    CUBE_SIZE,
    CUBE_MIDDLE,
    CUBE_LEN,
)
from cc2cc.utils.Grids import Grid
from cc2cc.utils.DataBase import DataBase
from cc2cc.utils.ModelDict import ModelDict
from cc2cc.utils.rotate import rotate


def save_csv_loss(
    name_list,
    path,
    dict_: dict,
):
    """
    save the loss to a csv file
    """
    dict_empty = {}
    dict_["name"] = name_list
    for key, val in dict_.items():
        dict_empty[key] = val
    df = pd.DataFrame(dict_empty)
    df.to_csv(path, index=False)


__all__ = [
    "gen_basis",
    "add_args",
    "Mol",
    "AU2KCALMOL",
    "AU2DEBYE",
    "MAIN_PATH",
    "DATA_PATH",
    "DATA_SCF_PATH",
    "DATA_TEST_PATH",
    "CUBE_SIZE",
    "CUBE_MIDDLE",
    "CUBE_LEN",
    "Grid",
    "DataBase",
    "ModelDict",
    "rotate",
    "extend",
    "save_csv_loss",
]
