"""Public utility API for cc2cc.

``Grid`` is the CPU/PySCF grid class by default.  GPU grid classes are loaded
lazily so importing ``cc2cc.utils`` does not require ``gpu4pyscf``/``cupy``.
"""

from cc2cc.utils.parser import add_args
from cc2cc.utils.mol import AU2DEBYE, AU2KCALMOL, gen_mole
from cc2cc.utils.env_var import DATA_PATH, MAIN_PATH, print_computer_info
from cc2cc.utils.TestDataDFT import TestDataDFT, diff_rho
from cc2cc.utils.timer import Timer
from cc2cc.utils.DataRecord import DataRecord
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.Grids import Grid, GridCPU

from cc2cc.utils.modelscf_rks import get_veff_modified_rks, get_veff_grad_modified_rks
from cc2cc.utils.modelscf_uks import get_veff_modified_uks, get_veff_grad_modified_uks

_GPU_EXPORTS = {"GridGPU"}

__all__ = [
    "add_args",
    "gen_mole",
    "get_veff_modified_rks",
    "get_veff_modified_uks",
    "get_veff_grad_modified_rks",
    "get_veff_grad_modified_uks",
    "print_computer_info",
    "diff_rho",
    "DataRecord",
    "TestDataDFT",
    "Timer",
    "Grid",
    "GridCPU",
    "ModelClass",
    "DATA_PATH",
    "MAIN_PATH",
    "AU2KCALMOL",
    "AU2DEBYE",
]


def _is_cuda_available():
    try:
        import cupy

        return True
    except ImportError:
        return False


def __getattr__(name):
    if name in _GPU_EXPORTS:
        if _is_cuda_available():
            from cc2cc.utils.GridsGPU import GridGPU
            from cc2cc.utils.modelscf_rks_gpu import (
                get_veff_modified_rks_gpu,
                get_veff_grad_modified_rks_gpu,
            )
            from cc2cc.utils.modelscf_uks_gpu import (
                get_veff_modified_uks_gpu,
                get_veff_grad_modified_uks_gpu,
            )

            globals().update(GridGPU=GridGPU)
            globals().update(get_veff_modified_rks_gpu=get_veff_modified_rks_gpu)
            globals().update(
                get_veff_grad_modified_rks_gpu=get_veff_grad_modified_rks_gpu
            )
            globals().update(get_veff_modified_uks_gpu=get_veff_modified_uks_gpu)
            globals().update(
                get_veff_grad_modified_uks_gpu=get_veff_grad_modified_uks_gpu
            )
            __all__.append("GridGPU")
            __all__.append("get_veff_modified_rks_gpu")
            __all__.append("get_veff_grad_modified_rks_gpu")
            __all__.append("get_veff_modified_uks_gpu")
            __all__.append("get_veff_grad_modified_uks_gpu")
            return globals()[name]
        raise ImportError(f"CUDA is not available. Cannot import {name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
