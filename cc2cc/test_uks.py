from timeit import default_timer as timer

import numpy as np

import pyscf

from cc2cc.utils import get_veff_modified_uks, diff_rho
from cc2cc.utils import AU2KCALMOL
from cc2cc.utils import TestData


def test_uks(
    mol,
    grids,
    name,
    modeldict,
    data_record,
    args,
):
    """
    Test the model. Restrict Khon-Sham (no spin).
    """
    # 2.0 Prepare
    test_data = TestData(
        mol,
        name,
        xc_code="b3lyp",
        if_grad=args.if_grad,
        cc_triple=args.cc_triple,
        if_disp=args.if_disp,
    )

    time_ai_start = timer()
    mdft = pyscf.dft.UKS(mol)
    mdft.xc = test_data.xc_code
    mdft.grids = grids
    mdft.verbose = 4

    get_veff_modified_uks(mdft, modeldict)

    if "test" in args.load:
        dm1_scf = test_data.dm1_cc.copy()
        e_scf = test_data.e_cc
    else:
        mdft.max_cycle = 50
        mdft.conv_tol = 1e-6
        if mol.natm == 1:
            # For single atom, use the dm from the test data
            mdft.kernel(dm0=test_data.mf_dm1)
        else:
            mdft.kernel()
        dm1_scf = mdft.make_rdm1()
        e_scf = mdft.e_tot

        # mdft.max_cycle = -1
        # mdft.kernel(dm0=test_data.dm1_cc)
        # dm1_scf = test_data.dm1_cc.copy()
        # e_scf = mdft.e_tot

    time_ai = timer() - time_ai_start

    # 3.0 Collect data
    scf_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_scf, unit="A.U.")
    error_dft_dip = np.linalg.norm(test_data.cc_dipole - test_data.dft_dipole)
    error_scf_dip = np.linalg.norm(test_data.cc_dipole - scf_dipole)

    error_scf_ele = diff_rho(mol, test_data.dm1_cc, dm1_scf, grids)
    error_dft_ele = diff_rho(mol, test_data.dm1_cc, test_data.dm1_dft, grids)

    dict_ = {
        "name": name,
        "cc_ene": AU2KCALMOL * test_data.e_cc,
        "scf_ene": AU2KCALMOL * e_scf,
        "dft_ene": AU2KCALMOL * test_data.e_dft,
        "error_scf_ele": error_scf_ele,
        "error_dft_ele": error_dft_ele,
        "error_scf_dip": error_scf_dip,
        "error_dft_dip": error_dft_dip,
        "time_cc": test_data.time_cc,
        "time_dft": test_data.time_dft,
        "time_ai": time_ai,
        "cc_dipole_x": test_data.cc_dipole[0],
        "cc_dipole_y": test_data.cc_dipole[1],
        "cc_dipole_z": test_data.cc_dipole[2],
        "scf_dipole_x": scf_dipole[0],
        "scf_dipole_y": scf_dipole[1],
        "scf_dipole_z": scf_dipole[2],
        "dft_dipole_x": test_data.dft_dipole[0],
        "dft_dipole_y": test_data.dft_dipole[1],
        "dft_dipole_z": test_data.dft_dipole[2],
    }
    if args.if_disp:
        dict_.update(
            {
                "delta_d3zero": AU2KCALMOL * test_data.delta_e["d3zero"],
                "delta_d3bj": AU2KCALMOL * test_data.delta_e["d3bj"],
            }
        )
    data_record.add_data(dict_)
    data_record.save_csv()
