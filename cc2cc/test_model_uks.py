from timeit import default_timer as timer

import numpy as np

import pyscf

from cc2cc.utils import get_veff_modified_uks, get_veff_grad_modified_uks, diff_rho
from cc2cc.utils import TestData, AU2KCALMOL


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
    mdft.xc = "b3lyp"
    mdft.grids = grids
    mdft.verbose = 4

    if modeldict.model_type == "center_4":
        get_veff_modified_uks(mdft, modeldict, max_memory=8000)
    elif modeldict.model_type == "cube":
        get_veff_modified_uks(mdft, modeldict, max_memory=800)

    if "test" in args.load:
        dm1_scf = test_data.dm1_dft.copy()
        e_scf = test_data.e_dft

        grad_mdft = None
    else:
        mdft.max_cycle = 50
        mdft.conv_tol = 1e-6
        if mol.natm == 1:
            # For single atom, use the dm from the test data
            mdft.kernel(dm0=test_data.dm1_dft)
        else:
            # mdft.kernel()
            mdft.kernel(dm0=test_data.dm1_dft)
        dm1_scf = mdft.make_rdm1()
        e_scf = mdft.e_tot

        if args.if_grad:
            g = mdft.Gradients()
            g.xc = test_data.xc_code
            g.grids = grids
            if modeldict.model_type == "center_4":
                get_veff_grad_modified_uks(
                    g,
                    modeldict,
                    max_memory=8000,
                    # dm_ks=test_data.dm1_dft,
                )
            elif modeldict.model_type == "cube":
                get_veff_grad_modified_uks(
                    g,
                    modeldict,
                    max_memory=800,
                    # dm_ks=test_data.dm1_dft,
                )
            grad_mdft = g.kernel()
        else:
            grad_mdft = None

    time_ai = timer() - time_ai_start

    # 3.0 Collect data
    scf_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_scf, unit="A.U.")

    dict_ = {
        "name": name,
        "delta_scf": AU2KCALMOL * (e_scf - test_data.e_cc),
        "delta_dft": AU2KCALMOL * (test_data.e_dft - test_data.e_cc),
        "time_cc": test_data.time_cc,
        "time_ai": time_ai,
        "time_dft": test_data.time_dft,
        "cc_ene": test_data.e_cc,
        "scf_ene": e_scf,
        "dft_ene": test_data.e_dft,
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
                "delta_d3zero": test_data.delta_e["d3zero"],
                "delta_d3bj": test_data.delta_e["d3bj"],
            }
        )
    if args.if_grad:
        if (
            grad_mdft is not None
            and test_data.grad_ccsd is not None
            and test_data.grad_dft is not None
        ):
            dict_.update(
                {
                    "delta_grad_scf": np.linalg.norm(grad_mdft - test_data.grad_ccsd),
                    "delta_grad_dft": np.linalg.norm(
                        test_data.grad_dft - test_data.grad_ccsd
                    ),
                }
            )
        else:
            dict_.update(
                {
                    "delta_grad_scf": 0,
                    "delta_grad_dft": 0,
                }
            )
    data_record.add_data(dict_)
    data_record.save_csv()
