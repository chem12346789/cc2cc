from timeit import default_timer as timer
import types

import numpy as np
import opt_einsum as oe

import pyscf
from pyscf import lib

from cc2cc.utils import DATA_PATH, AU2KCALMOL
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
    density_restriction = getattr(args, "density_restriction", 0)
    if_grad = getattr(args, "if_grad", False)
    cc_triple = getattr(args, "cc_triple", False)

    # 2.0 Prepare
    test_data = TestData(
        mol,
        name,
        xc_code="b3lyp",
        if_grad=if_grad,
        cc_triple=cc_triple,
    )

    time_dft_start = timer()
    mdft_dft = pyscf.dft.RKS(mol)
    mdft_dft.xc = test_data.xc_code
    # mdft_dft.grids.atom_grid = {
    #     "C": (75, 302),
    #     "O": (75, 302),
    #     "N": (75, 302),
    #     "H": (75, 302),
    # }
    mdft_dft.kernel(dm0=test_data.mf_dm1)
    test_data.e_dft = mdft_dft.e_tot
    test_data.dm1_dft = mdft_dft.make_rdm1()
    test_data.time_dft = timer() - time_dft_start

    time_ai_start = timer()

    mdft = pyscf.dft.RKS(mol)
    mdft.xc = test_data.xc_code
    mdft.grids = grids

    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
    ao_0 = ao_value[0]

    def get_veff_modified(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks.mol

        if dm is None:
            dm = ks.make_rdm1()

        ground_state = isinstance(dm, np.ndarray) and dm.ndim == 2
        ni = ks._numint

        nelec, exc, vxc = modeldict.nr_rks(ni, mol, grids, dm, test_data.xc_code)

        if not ni.libxc.is_hybrid_xc(ks.xc):
            vk = None
            if (
                ks._eri is None
                and ks.direct_scf
                and getattr(vhf_last, "vj", None) is not None
            ):
                ddm = np.asarray(dm) - np.asarray(dm_last)
                vj = ks.get_j(mol, ddm, hermi)
                vj += vhf_last.vj
            else:
                vj = ks.get_j(mol, dm, hermi)
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
            if (
                ks._eri is None
                and ks.direct_scf
                and getattr(vhf_last, "vk", None) is not None
            ):
                ddm = np.asarray(dm) - np.asarray(dm_last)
                vj, vk = ks.get_jk(mol, ddm, hermi)
                vk *= hyb
                if omega != 0:  # For range separated Coulomb
                    vklr = ks.get_k(mol, ddm, hermi, omega=omega)
                    vklr *= alpha - hyb
                    vk += vklr
                vj += vhf_last.vj
                vk += vhf_last.vk
            else:
                vj, vk = ks.get_jk(mol, dm, hermi)
                vk *= hyb
                if omega != 0:
                    vklr = ks.get_k(mol, dm, hermi, omega=omega)
                    vklr *= alpha - hyb
                    vk += vklr
            vxc += vj - vk * 0.5

            if ground_state:
                exc -= np.einsum("ij,ji", dm, vk).real * 0.5 * 0.5

        if ground_state:
            ecoul = np.einsum("ij,ji", dm, vj).real * 0.5
        else:
            ecoul = None

        vxc = lib.tag_array(
            vxc,
            ecoul=ecoul,
            exc=exc,
            vj=vj,
            vk=vk,
        )

        return vxc

    mdft.get_veff = types.MethodType(get_veff_modified, mdft)

    mdft.max_cycle = 50
    mdft.conv_tol = 1e-5
    mdft.diis_space = 10
    mdft.kernel(dm0=test_data.mf_dm1)
    dm1_scf = mdft.make_rdm1()

    # mdft.max_cycle = -1
    # mdft.kernel(dm0=test_data.dm1_cc)
    # dm1_scf = test_data.dm1_cc.copy()

    scf_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_scf, unit="A.U.")

    time_ai = timer() - time_ai_start

    error_dft_ene = AU2KCALMOL * (test_data.e_cc - test_data.e_dft)
    error_scf_ene = AU2KCALMOL * (test_data.e_cc - mdft.e_tot)
    error_dft_dip = np.linalg.norm(test_data.cc_dipole - test_data.dft_dipole)
    error_scf_dip = np.linalg.norm(test_data.cc_dipole - scf_dipole)

    rho_scf = pyscf.dft.numint.eval_rho(
        mol,
        ao_value[0],
        dm1_scf,
        xctype="LDA",
    )
    rho_cc = pyscf.dft.numint.eval_rho(
        mol,
        ao_value[0],
        test_data.dm1_cc,
        xctype="LDA",
    )
    rho_dft = pyscf.dft.numint.eval_rho(
        mol,
        ao_value[0],
        test_data.dm1_dft,
        xctype="LDA",
    )
    error_scf_ele = np.sum(np.abs(rho_cc - rho_scf) * grids.weights)
    error_dft_ele = np.sum(np.abs(rho_cc - rho_dft) * grids.weights)

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
