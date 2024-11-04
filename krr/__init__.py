from functools import reduce

import numpy as np

from krr.argparse import add_args
from krr.load_data import load_data
from krr.krr import hash_value, evaluate, add_data
from krr.krr import KernelRidgeModified


def append(*args):
    list_ = [i for i in args]
    return reduce(np.append, list_)
