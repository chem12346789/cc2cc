"""Test the model. Restrict Khon-Sham (no spin)."""

from timeit import default_timer as timer

import pyscf

from cc2cc.utils import get_veff_modified_rks, get_veff_grad_modified_rks


def test_model_rks(
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
    time_ai_start = timer()
    mdft = pyscf.dft.RKS(mol).density_fit()
    mdft.xc = "b3lyp"
    mdft.grids = grids
    mdft.verbose = 4
    mdft.max_cycle = 50
    mdft.conv_tol = 1e-7

    if modeldict.model_type == "center_4":
        get_veff_modified_rks(mdft, modeldict, max_memory=8000)
    elif modeldict.model_type == "cube":
        get_veff_modified_rks(mdft, modeldict, max_memory=800)

    mdft.kernel()
    # mdft.kernel(dm0=test_data.dm1_dft)
    if mdft.converged is False:
        print("RKS not converged. First try.")
        mdft.diis_damp = 0.5
        mdft.kernel()
        if mdft.converged is False:
            print("RKS not converged. Second try.")
            mdft.conv_tol = 1e-6
            mdft.diis_damp = 0.0
            pyscf.scf.addons.dynamic_level_shift_(mdft, factor=0.5)
            mdft.kernel()
            if mdft.converged is False:
                print("RKS not converged. Third try.")
                mdft.level_shift = 0.0
                mdft = mdft.newton()
                mdft.kernel()
                if mdft.converged is False:
                    raise ValueError("RKS not converged.")
    dm1_scf = mdft.make_rdm1()
    e_scf = mdft.e_tot

    if args.if_grad:
        g = mdft.Gradients()
        g.xc = "b3lyp"
        g.grids = grids
        if modeldict.model_type == "center_4":
            get_veff_grad_modified_rks(
                g,
                modeldict,
                max_memory=8000,
                # dm_ks=test_data.dm1_dft,
            )
        elif modeldict.model_type == "cube":
            get_veff_grad_modified_rks(
                g,
                modeldict,
                max_memory=800,
                # dm_ks=test_data.dm1_dft,
            )
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

    data_record.add_data(dict_)
    data_record.save_csv()
