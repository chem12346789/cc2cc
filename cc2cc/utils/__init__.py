import numpy as np
from pyscf import dft

from cc2cc.utils.parser import add_args
from cc2cc.utils.mol import gen_mole

from cc2cc.utils.Grids import Grid
from cc2cc.utils.DataBaseCenter import DataBaseCenter
from cc2cc.utils.DataBaseCube import DataBaseCube
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.modelscf_rks import get_veff_modified as get_veff_modified_rks
from cc2cc.utils.modelscf_uks import get_veff_modified as get_veff_modified_uks
from cc2cc.utils.rotate import rotate
from cc2cc.utils.DataRecord import DataRecord
from cc2cc.utils.TestData import TestData

from cc2cc.utils.env_var import (
    print_gpu_info,
    MAIN_PATH,
    DATA_PATH,
    DATA_SCF_PATH,
    DATA_TEST_PATH,
    CUBE_SIZE,
    CUBE_MIDDLE,
    CUBE_LEN,
    TEST,
    GENERATE_DATA,
)
from cc2cc.utils.mol import AU2KCALMOL, AU2DEBYE


def diff_rho(mol, dm1_compare1, dm1_compare2, grids):
    """
    Calculate the difference between two density matrices.
    """
    ao = dft.numint.eval_ao(mol, grids.coords, deriv=0)
    if len(np.shape(dm1_compare1)) != len(np.shape(dm1_compare2)):
        raise ValueError("dm1_compare1 and dm1_compare2 must have the same dimension.")
    if len(np.shape(dm1_compare1)) == 3:
        dm1_compare1 = dm1_compare1[0] + dm1_compare1[1]
        dm1_compare2 = dm1_compare2[0] + dm1_compare2[1]
    ddm = dm1_compare1 - dm1_compare2
    drho = dft.numint.eval_rho(mol, ao, ddm, xctype="LDA")

    return np.sum(np.abs(drho) * grids.weights)


__all__ = [
    "add_args",
    "gen_mole",
    "Grid",
    "DataBaseCenter",
    "DataBaseCube",
    "ModelClass",
    "get_veff_modified_rks",
    "get_veff_modified_uks",
    "rotate",
    "print_gpu_info",
    "DataRecord",
    "TestData",
    "MAIN_PATH",
    "DATA_PATH",
    "DATA_SCF_PATH",
    "DATA_TEST_PATH",
    "CUBE_SIZE",
    "CUBE_MIDDLE",
    "CUBE_LEN",
    "TEST",
    "GENERATE_DATA",
    "AU2KCALMOL",
    "AU2DEBYE",
]
