"""Test the model. Unrestricted Khon-Sham (with spin)."""

import numpy as np
from timeit import default_timer as timer
import torch

import pyscf
import pyscf.dft

import cc2cc.utils as utils

pyscf.lib.logger.TIMER_LEVEL = 4


def test_model_uks(
    mol,
    name,
    modeldict,
    data_record,
    args,
):
    """
    Test the model. Unrestricted Khon-Sham (with spin).
    """
    # 2.0 Prepare
    time_ai_start = timer()
    if torch.cuda.is_available():
        mdft = pyscf.dft.UKS(mol).density_fit().to_gpu()
        Grid = utils.GridGPU
        mol.stdout = mdft.stdout
        mdft.max_memory = 8000
        utils.get_veff_modified_uks_gpu(mdft, modeldict)
    else:
        mdft = pyscf.dft.UKS(mol).density_fit()
        Grid = utils.GridCPU
        utils.get_veff_modified_uks(mdft, modeldict)

    mdft.xc = "b3lyp"
    mdft.grids = Grid(
        mol,
        args.grid_level,
        input_level=modeldict.input_level,
        cube_type=modeldict.cube_type,
        cube_size=modeldict.cube_size,
    )

    mdft.verbose = 4
    mdft.mol.verbose = 4
    mdft.conv_tol = 1e-7
    mdft.conv_tol_grad = 1e-3

    if args.max_cycle == -1:
        mdft.max_cycle = -1
        if_retry = False
        test_data = utils.TestDataDFT(mol, name, xc_code=mdft.xc, disp=None)
        mdft.kernel(dm0=test_data.dm1_dft)
    else:
        # mdft.max_cycle = args.max_cycle
        mdft.max_cycle = 200
        if_retry = True
        mdft.kernel()

    if mdft.converged is False and if_retry:
        print("UKS not converged. Add dynamic level shift.")
        pyscf.scf.addons.dynamic_level_shift_(mdft, factor=1.0)
        mdft.kernel()
    if mdft.converged is False and if_retry:
        print("UKS not converged. Add dynamic level shift.")
        pyscf.scf.addons.dynamic_level_shift_(mdft, factor=2.0)
        mdft.kernel()
    if mdft.converged is False and if_retry:
        print("UKS not converged. Add dynamic level shift.")
        pyscf.scf.addons.dynamic_level_shift_(mdft, factor=4.0)
        mdft.kernel()
    if mdft.converged is False and if_retry:
        print("UKS not converged. Add dynamic level shift.")
        pyscf.scf.addons.dynamic_level_shift_(mdft, factor=8.0)
        mdft.kernel()
    if mdft.converged is False:
        print("Error: UKS not converged!!! Just use the current result.")
    dm1_scf = mdft.make_rdm1()
    e_scf = mdft.e_tot

    if args.if_grad and args.max_cycle != -1:
        g = mdft.Gradients()
        g.xc = "b3lyp"
        g.grids = mdft.grids
        utils.get_veff_grad_modified_uks(g, modeldict)
        grad_mdft = g.kernel()
    else:
        grad_mdft = None

    time_ai = timer() - time_ai_start

    # 3.0 Collect data

    if torch.cuda.is_available():
        import cupy as cp

        dm1_scf = cp.asnumpy(dm1_scf)
    scf_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_scf, unit="A.U.")

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

    if (utils.DATA_PATH / f"data_{name}.npz").exists():
        data = np.load(utils.DATA_PATH / f"data_{name}.npz", allow_pickle=True)
        print(data["mol"])
        print(mol.tostring(format="xyz"))
        print(
            f"electronic density (ai vs dft) {utils.diff_rho(mol, data["dm1_dft"], mdft.make_rdm1(), mdft.grids)}"
        )
        print(
            f"electronic density (ai vs cc) {utils.diff_rho(mol, data["dm1_cc"], mdft.make_rdm1(), mdft.grids)}"
        )
        print(
            f"electronic density (dft vs cc) {utils.diff_rho(mol, data["dm1_cc"], data["dm1_dft"], mdft.grids)}"
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
