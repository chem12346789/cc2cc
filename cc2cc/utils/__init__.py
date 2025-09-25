import numpy as np
from pyscf import dft

from cc2cc.utils.parser import add_args
from cc2cc.utils.mol import gen_mole
from cc2cc.utils.modelscf_rks import get_veff_modified as get_veff_modified_rks
from cc2cc.utils.modelscf_rks import (
    get_veff_grad_modified as get_veff_grad_modified_rks,
)
from cc2cc.utils.modelscf_uks import get_veff_modified as get_veff_modified_uks
from cc2cc.utils.modelscf_uks import (
    get_veff_grad_modified as get_veff_grad_modified_uks,
)
from cc2cc.utils.env_var import print_computer_info
from cc2cc.utils.TestDataDFT import diff_rho

from cc2cc.utils.Grids import Grid
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.DataRecord import DataRecord
from cc2cc.utils.TestDataDFT import TestDataDFT
from cc2cc.utils.timer import Timer

from cc2cc.utils.env_var import DATA_PATH
from cc2cc.utils.mol import AU2KCALMOL, AU2DEBYE


__all__ = [
    "add_args",
    "gen_mole",
    "get_veff_modified_rks",
    "get_veff_modified_uks",
    "print_computer_info",
    "diff_rho",
    "DataRecord",
    "TestDataDFT",
    "Timer",
    "Grid",
    "ModelClass",
    "DATA_PATH",
    "AU2KCALMOL",
    "AU2DEBYE",
]
