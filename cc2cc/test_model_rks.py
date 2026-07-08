"""Test the model. Restrict Khon-Sham (no spin)."""

import timeit
import gc
import numpy as np
import torch

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

import pyscf
import pyscf.dft

import cc2cc.utils as utils

pyscf.lib.logger.TIMER_LEVEL = 4


def _release_memory(device) -> None:
    gc.collect()
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
    time_ai_start = timeit.default_timer()
    if torch.cuda.is_available():
        print("Use GPU for DFT calculation.")
        mdft = pyscf.dft.RKS(mol).to_gpu().density_fit()
        Grid = utils.GridGPU
        utils.get_veff_modified_rks_gpu(mdft, modeldict)
        mol.stdout = mdft.stdout
        mdft.use_gpu_memory = False
    else:
        print("Use CPU for DFT calculation.")
        Grid = utils.GridCPU
        mdft = pyscf.dft.RKS(mol).density_fit()
        utils.get_veff_modified_rks(mdft, modeldict)

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
        mdft.max_cycle = args.max_cycle
        if_retry = True
        mdft.kernel()

    for factor in [1.0, 2.0, 4.0, 8.0]:
        if mdft.converged is False and if_retry:
            print(f"RKS not converged. Add dynamic level shift with factor {factor}.")
            pyscf.scf.addons.dynamic_level_shift_(mdft, factor=factor)
            mdft.kernel()
    if mdft.converged is False:
        print("Error: RKS not converged!!! Just use the current result.")

    dm1_scf = mdft.make_rdm1()
    e_scf = mdft.e_tot

    if args.if_grad and args.max_cycle != -1:
        g = mdft.Gradients()
        g.xc = "b3lyp"
        g.grids = mdft.grids
        if torch.cuda.is_available():
            utils.get_veff_grad_modified_rks_gpu(g, modeldict)
        else:
            utils.get_veff_grad_modified_rks(g, modeldict)
        grad_mdft = g.kernel()
    else:
        grad_mdft = None

    time_ai = timeit.default_timer() - time_ai_start

    # 3.0 Collect data

    if torch.cuda.is_available():
        _release_memory(args.device)
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
            f"energy (ai vs dft) {(e_scf - data['e_dft']) * utils.AU2KCALMOL} Kcal/mol"
        )
        print(f"energy (ai vs cc) {(e_scf - data['e_cc']) * utils.AU2KCALMOL} Kcal/mol")

        dm1_dft = data["dm1_dft"]
        dm1_cc = data["dm1_cc"]
        dm1_ai = dm1_scf
        print(
            f"electronic density (ai vs dft) {utils.diff_rho(mol, dm1_dft, dm1_ai, mdft.grids)}"
        )
        print(
            f"electronic density (ai vs cc) {utils.diff_rho(mol, dm1_cc, dm1_ai, mdft.grids)}"
        )
        print(
            f"electronic density (dft vs cc) {utils.diff_rho(mol, dm1_cc, dm1_dft, mdft.grids)}"
        )
        dipole_dft = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
        dipole_cc = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")

        print(f"dipole (ai vs dft) {np.linalg.norm(scf_dipole - dipole_dft)}")
        print(f"dipole (ai vs cc) {np.linalg.norm(scf_dipole - dipole_cc)}")
        print(f"dipole (dft vs cc) {np.linalg.norm(dipole_dft - dipole_cc)}")
        grad_dft = data["grad_dft"]
        grad_cc = data["grad_cc"]
        if args.if_grad:
            print(f"gradient (ai vs dft) {np.linalg.norm(grad_mdft - grad_dft)}")
            print(f"gradient (ai vs cc) {np.linalg.norm(grad_mdft - grad_cc)}")
            print(f"gradient (dft vs cc) {np.linalg.norm(grad_dft - grad_cc)}")

    data_record.add_data(dict_)
    data_record.save_csv()
