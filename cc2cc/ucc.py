import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
import opt_einsum as oe

from cc2cc.utils import Grid
from cc2cc.utils import DATA_PATH, AU2KCALMOL


def cc(mol, name):