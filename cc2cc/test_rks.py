from timeit import default_timer as timer

import numpy as np

import pyscf

from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils import Grid, Test_Data


def test_rks(
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
    mdft = pyscf.dft.RKS(mol)

    time_ai_start = timer()
    data = np.load(DATA_PATH / f"data_{name}.npz")
    exc_over_rho_cc_grids = data["exc_over_dm_cc_grids"]
    rho_cc, exc_over_rho_cc_grids_pred = modeldict.get_e_density(
        rks=mdft,
        grids=grids,
        dms=test_data.dm1_cc,
    )
    time_ai = timer() - time_ai_start

    weights = grids.weights
    print(np.linalg.norm(weights - data["weights"]))

    error_scf_ene = AU2KCALMOL * np.sum(
        (exc_over_rho_cc_grids_pred - exc_over_rho_cc_grids) * rho_cc * weights
    )
    abs_error_scf_ene = AU2KCALMOL * np.sum(
        np.abs((exc_over_rho_cc_grids_pred - exc_over_rho_cc_grids) * rho_cc * weights)
    )
    error_dft_ene = AU2KCALMOL * data["error_energy"]
    print(error_scf_ene)
    print(abs_error_scf_ene)

    print(error_dft_ene)
    print(AU2KCALMOL * np.sum(exc_over_rho_cc_grids * rho_cc * weights))

    data_record.add_data(
        name,
        {
            "error_scf_ene": error_scf_ene,
            "error_dft_ene": error_dft_ene,
            "abs_error_scf_ene": abs_error_scf_ene,
            "time_cc": test_data.time_cc,
            "time_dft": test_data.time_dft,
            "time_ai": time_ai,
        },
    )
    data_record.save_csv()
