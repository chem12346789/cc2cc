from timeit import default_timer as timer

import numpy as np

import pyscf

from cc2cc.utils import get_veff_modified_rks
from cc2cc.utils import AU2KCALMOL
from cc2cc.utils import TestData


def test_rks(
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
        disp=args.disp,
    )

    time_ai_start = timer()
    mdft = pyscf.dft.RKS(mol)
    mdft.xc = test_data.xc_code
    mdft.disp = test_data.disp
    mdft.grids = grids
    mdft.verbose = 4

    get_veff_modified_rks(mdft, modeldict)

    if "test" in args.load:
        dm1_scf = test_data.dm1_cc.copy()
        e_scf = test_data.e_cc
    else:
        mdft.max_cycle = 50
        mdft.conv_tol = 1e-6
        mdft.kernel()
        dm1_scf = mdft.make_rdm1()
        e_scf = mdft.e_tot

        # mdft.max_cycle = -1
        # mdft.kernel(dm0=test_data.dm1_cc)
        # dm1_scf = test_data.dm1_cc.copy()
        # e_scf = mdft.e_tot

    time_ai = timer() - time_ai_start

    error_dft_ene = AU2KCALMOL * (test_data.e_cc - test_data.e_dft)
    error_scf_ene = AU2KCALMOL * (test_data.e_cc - e_scf)

    scf_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_scf, unit="A.U.")
    error_dft_dip = np.linalg.norm(test_data.cc_dipole - test_data.dft_dipole)
    error_scf_dip = np.linalg.norm(test_data.cc_dipole - scf_dipole)

    ao = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=0)
    rho_scf = pyscf.dft.numint.eval_rho(mol, ao, dm1_scf, xctype="LDA")
    rho_cc = pyscf.dft.numint.eval_rho(mol, ao, test_data.dm1_cc, xctype="LDA")
    rho_dft = pyscf.dft.numint.eval_rho(mol, ao, test_data.dm1_dft, xctype="LDA")
    error_scf_ele = np.sum(np.abs(rho_cc - rho_scf) * grids.weights)
    error_dft_ele = np.sum(np.abs(rho_cc - rho_dft) * grids.weights)

    data_record.add_data(
        {
            "name": name,
            "error_scf_ene": error_scf_ene,
            "error_dft_ene": error_dft_ene,
            "error_scf_ele": error_scf_ele,
            "error_dft_ele": error_dft_ele,
            "error_scf_dip": error_scf_dip,
            "error_dft_dip": error_dft_dip,
            "time_cc": test_data.time_cc,
            "time_dft": test_data.time_dft,
            "time_ai": time_ai,
            "cc_ene": AU2KCALMOL * test_data.e_cc,
            "scf_ene": AU2KCALMOL * e_scf,
            "dft_ene": AU2KCALMOL * test_data.e_dft,
            "cc_dipole_x": test_data.cc_dipole[0],
            "cc_dipole_y": test_data.cc_dipole[1],
            "cc_dipole_z": test_data.cc_dipole[2],
            "scf_dipole_x": scf_dipole[0],
            "scf_dipole_y": scf_dipole[1],
            "scf_dipole_z": scf_dipole[2],
            "dft_dipole_x": test_data.dft_dipole[0],
            "dft_dipole_y": test_data.dft_dipole[1],
            "dft_dipole_z": test_data.dft_dipole[2],
        },
    )
    data_record.save_csv()
