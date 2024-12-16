from pathlib import Path

import pandas as pd
import numpy as np

import pyscf

from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils import Grid, Test_Data


def test_uks(
    mol,
    name,
    modeldict,
    data_record,
):
    """
    Test the model. Restrict Khon-Sham (no spin).
    """
    # 2.0 Prepare
    test_data = Test_Data(mol, name)
    test_data.test_mol()
    grids = Grid(test_data.mol)

    mdft = pyscf.dft.UKS(mol)

    data = np.load(DATA_PATH / f"data_{name}.npz")
    exc_over_rho_cc_grids_pred = modeldict.get_e_density(
        rks=mdft,
        grids=grids,
        dms=test_data.dm1_cc,
    )
    weights = grids.weights
    rho_cc, exc_over_rho_cc_grids = data["exc_over_dm_cc_grids"]

    print(np.linalg.norm(weights - data["weights"]))
    print(
        AU2KCALMOL
        * np.sum(
            (exc_over_rho_cc_grids_pred - exc_over_rho_cc_grids) * rho_cc * weights
        )
    )
    print(
        AU2KCALMOL
        * np.sum(
            np.abs(
                (exc_over_rho_cc_grids_pred - exc_over_rho_cc_grids) * rho_cc * weights
            )
        )
    )

    print(AU2KCALMOL * data["error_energy"])
    print(AU2KCALMOL * np.sum(exc_over_rho_cc_grids * rho_cc * weights))
