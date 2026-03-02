"""Test the model. Restrict Khon-Sham (no spin)."""

import numpy as np
from timeit import default_timer as timer

import pyscf
import pyscf.dft

from cc2cc.utils import get_veff_modified_rks, get_veff_grad_modified_rks, diff_rho
from cc2cc.utils import Grid, TestDataDFT, DATA_PATH


def test_model_rks(
    mol,
    name,
    modeldict,
    data_record,
    args,
):
    """
    Test the model. Restrict Khon-Sham (no spin).
    """
    # 2.0 Prepare
    time_ai_start = timer()
    mdft = pyscf.dft.RKS(mol).density_fit()
    mdft.xc = "b3lyp"
    mdft.grids = Grid(mol, args.grid_level, modeldict.input_level, test=True)

    mdft.verbose = 4
    mdft.mol.verbose = 4
    mdft.diis_space = 6
    mdft.conv_tol = 1e-7
    mdft.conv_tol_grad = 1e-3

    if modeldict.model_type == "center_4":
        get_veff_modified_rks(mdft, modeldict)
    elif modeldict.model_type == "cube":
        get_veff_modified_rks(mdft, modeldict)

    if args.max_cycle == -1:
        mdft.max_cycle = -1
        if_retry = False
        test_data = TestDataDFT(mol, name, xc_code=mdft.xc, disp=None)
        mdft.kernel(dm0=test_data.dm1_dft)
    else:
        mdft.max_cycle = args.max_cycle
        if_retry = True
        mdft.kernel()

    if mdft.converged is False and if_retry:
        print("RKS not converged. Add dynamic level shift.")
        pyscf.scf.addons.dynamic_level_shift_(mdft, factor=0.5)
        mdft.kernel()
        if mdft.converged is False:
            print("Error: RKS not converged!!! Restart without SCF procedure.")
            test_data = TestDataDFT(mol, name, xc_code=mdft.xc, disp=None)
            mdft.max_cycle = -1
            mdft.kernel(dm0=test_data.dm1_dft)
    dm1_scf = mdft.make_rdm1()
    e_scf = mdft.e_tot

    if args.if_grad and args.max_cycle != -1:
        g = mdft.Gradients()
        g.xc = "b3lyp"
        g.grids = mdft.grids
        if modeldict.model_type == "center_4":
            get_veff_grad_modified_rks(g, modeldict)
        elif modeldict.model_type == "cube":
            get_veff_grad_modified_rks(g, modeldict)
        grad_mdft = g.kernel()
    else:
        grad_mdft = None

    scf_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_scf, unit="A.U.")
    time_ai = timer() - time_ai_start

    # 3.0 Collect data
    dict_ = {
        "name": name,
        "time_ai": time_ai,
        "scf_ene": e_scf,
        "scf_dipole_x": scf_dipole[0],
        "scf_dipole_y": scf_dipole[1],
        "scf_dipole_z": scf_dipole[2],
    }

    if args.if_grad:
        if grad_mdft is not None:
            dict_.update({"grad_scf": grad_mdft})
        else:
            dict_.update({"grad_scf": 0})

    if (DATA_PATH / f"data_{name}.npz").exists():
        data = np.load(DATA_PATH / f"data_{name}.npz", allow_pickle=True)
        print(data["mol"])
        print(mol.tostring(format="xyz"))
        print(
            f"electronic density (ai vs dft) {diff_rho(mol, data["dm1_dft"], mdft.make_rdm1(), mdft.grids)}"
        )
        print(
            f"electronic density (ai vs cc) {diff_rho(mol, data["dm1_cc"], mdft.make_rdm1(), mdft.grids)}"
        )
        print(
            f"electronic density (dft vs cc) {diff_rho(mol, data["dm1_cc"], data["dm1_dft"], mdft.grids)}"
        )
        print(f"energy (ai vs dft) {(e_scf - data['e_dft']) * 627.509} Kcal/mol")
        print(f"energy (ai vs cc) {(e_scf - data['e_cc']) * 627.509} Kcal/mol")
        if args.if_grad:
            print(
                f"gradient (ai vs dft) {np.linalg.norm(grad_mdft - data['grad_dft'])}"
            )
            print(f"gradient (ai vs cc) {np.linalg.norm(grad_mdft - data['grad_cc'])}")
            print(
                f"gradient (dft vs cc) {np.linalg.norm(data['grad_dft'] - data['grad_cc'])}"
            )

    data_record.add_data(dict_)
    data_record.save_csv()
