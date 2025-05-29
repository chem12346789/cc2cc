from timeit import default_timer as timer

import numpy as np

import pyscf

from cc2cc.utils import get_veff_modified_uks
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
    density_restriction = getattr(args, "density_restriction", 0)
    if_grad = getattr(args, "if_grad", False)
    cc_triple = getattr(args, "cc_triple", False)
    use_orca = getattr(args, "use_orca", False)

    # 2.0 Prepare
    test_data = TestData(
        mol,
        name,
        xc_code="b3lyp",
        if_grad=if_grad,
        cc_triple=cc_triple,
        use_orca=use_orca,
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
        mdft.conv_tol = 1e-5
        mdft.kernel(dm0=test_data.mf_dm1)
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
    rho_scf = [
        pyscf.dft.numint.eval_rho(mol, ao, dm1_scf[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao, dm1_scf[1], xctype="LDA"),
    ]
    rho_cc = [
        pyscf.dft.numint.eval_rho(mol, ao, test_data.dm1_cc[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao, test_data.dm1_cc[1], xctype="LDA"),
    ]
    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao, test_data.dm1_dft[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao, test_data.dm1_dft[1], xctype="LDA"),
    ]
    error_scf_ele = np.sum(
        np.abs(rho_cc[0] - rho_scf[0]) * grids.weights
        + np.abs(rho_cc[1] - rho_scf[1]) * grids.weights
    )
    error_dft_ele = np.sum(
        np.abs(rho_cc[0] - rho_dft[0]) * grids.weights
        + np.abs(rho_cc[1] - rho_dft[1]) * grids.weights
    )

    data_record.add_data(
        name,
        {
            "error_scf_ene": error_scf_ene,
            "error_dft_ene": error_dft_ene,
            "error_scf_ele": error_scf_ele,
            "error_dft_ele": error_dft_ele,
            "error_scf_dip": error_scf_dip,
            "error_dft_dip": error_dft_dip,
            "time_cc": test_data.time_cc,
            "time_dft": test_data.time_dft,
            "time_ai": time_ai,
        },
    )
    data_record.save_csv()
