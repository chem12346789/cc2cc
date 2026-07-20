from typing import Any

from cc2cc.utils.DataRecord import DataRecord as DataRecord
from cc2cc.utils.Grids import Grid as Grid
from cc2cc.utils.Grids import GridCPU as GridCPU
from cc2cc.utils.GridsGPU import GridGPU as GridGPU
from cc2cc.utils.ModelClass import ModelClass as ModelClass
from cc2cc.utils.TestDataDFT import TestDataDFT as TestDataDFT
from cc2cc.utils.TestDataDFT import diff_rho as diff_rho
from cc2cc.utils.computer_info import print_computer_info as print_computer_info
from cc2cc.utils.env_var import DATA_PATH as DATA_PATH
from cc2cc.utils.env_var import MAIN_PATH as MAIN_PATH
from cc2cc.utils.modelscf_rks import (
    get_veff_grad_modified_rks as get_veff_grad_modified_rks,
)
from cc2cc.utils.modelscf_rks import get_veff_modified_rks as get_veff_modified_rks
from cc2cc.utils.modelscf_rks_gpu import (
    get_veff_grad_modified_rks_gpu as get_veff_grad_modified_rks_gpu,
)
from cc2cc.utils.modelscf_rks_gpu import (
    get_veff_modified_rks_gpu as get_veff_modified_rks_gpu,
)
from cc2cc.utils.modelscf_uks import (
    get_veff_grad_modified_uks as get_veff_grad_modified_uks,
)
from cc2cc.utils.modelscf_uks import get_veff_modified_uks as get_veff_modified_uks
from cc2cc.utils.modelscf_uks_gpu import (
    get_veff_grad_modified_uks_gpu as get_veff_grad_modified_uks_gpu,
)
from cc2cc.utils.modelscf_uks_gpu import (
    get_veff_modified_uks_gpu as get_veff_modified_uks_gpu,
)
from cc2cc.utils.mol import AU2DEBYE as AU2DEBYE
from cc2cc.utils.mol import AU2KCALMOL as AU2KCALMOL
from cc2cc.utils.mol import gen_mole as gen_mole
from cc2cc.utils.parser import add_args as add_args
from cc2cc.utils.parser import config_list as config_list
from cc2cc.utils.parser import process_config as process_config
from cc2cc.utils.timer import Timer as Timer

NO_LAZY_EXPORTS: dict[str, tuple[str, str]]
_LAZY_EXPORTS: dict[str, tuple[str, str]]
_GPU_EXPORTS: dict[str, tuple[str, str]]
__all__: list[str]

def _load_symbol(name: str, table: dict[str, tuple[str, str]]) -> Any: ...
def __getattr__(name: str) -> Any: ...
def __dir__() -> list[str]: ...
