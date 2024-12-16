from timeit import default_timer as timer
import types

import numpy as np
import torch

import pyscf
from pyscf import lib

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
    test_data = Test_Data(mol, name, xc_code="b3lyp")
    test_data.test_mol()
    grids = Grid(test_data.mol)
    mdft = pyscf.dft.RKS(mol)
    mdft.xc = test_data.xc_code
    mdft.grids = grids

    time_ai_start = timer()

    def get_veff_modified(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks.mol

        if dm is None:
            dm = ks.make_rdm1()

        ground_state = isinstance(dm, np.ndarray) and dm.ndim == 2
        ni = ks._numint

        nelec, excsum, vmat = modeldict.get_nev(ni, ks, grids, dm, test_data.xc_code)

        # rho_diff = ni.eval_rho(dft2cc.mol, dft2cc.ao_0, dm - dft2cc.dm1_cc)
        # v_p = pyscf.dft.numint.eval_mat(
        #     dft2cc.mol, dft2cc.ao_0, dft2cc.grids.weights, rho_diff, rho_diff
        # )
        # vxc += 100 * v_p

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
            vmat += vj
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
            vmat += vj - vk * 0.5

            if ground_state:
                excsum -= np.einsum("ij,ji", dm, vk).real * 0.5 * 0.5

        if ground_state:
            ecoul = np.einsum("ij,ji", dm, vj).real * 0.5
        else:
            ecoul = None

        vxc = lib.tag_array(
            vmat,
            ecoul=ecoul,
            exc=excsum,
            vj=vj,
            vk=vk,
        )

        return vxc

    mdft.get_veff = types.MethodType(get_veff_modified, mdft)
    mdft.conv_tol = 1e-6
    mdft.kernel()

    scf_dipole = pyscf.scf.hf.dip_moment(
        mol=mol,
        dm=mdft.make_rdm1(),
        unit="A.U.",
    )

    # e_scf = modeldict.get_e(mdft, grids, test_data.dm1_cc)
    # print(AU2KCALMOL * (test_data.e_cc - mdft.energy_tot(test_data.dm1_cc)))
    # print(AU2KCALMOL * e_scf)
    # e_scf += mdft.energy_tot(test_data.dm1_cc)

    time_ai = timer() - time_ai_start

    error_dft_ene = AU2KCALMOL * (test_data.e_dft - test_data.e_cc)
    error_scf_ene = AU2KCALMOL * (mdft.e_tot - test_data.e_cc)
    error_dft_dip = np.linalg.norm(test_data.dft_dipole - test_data.cc_dipole)
    error_scf_dip = np.linalg.norm(scf_dipole - test_data.cc_dipole)

    data_record.add_data(
        name,
        {
            "error_scf_ene": error_scf_ene,
            "error_dft_ene": error_dft_ene,
            "error_scf_dip": error_scf_dip,
            "error_dft_dip": error_dft_dip,
            "time_cc": test_data.time_cc,
            "time_dft": test_data.time_dft,
            "time_ai": time_ai,
        },
    )
    data_record.save_csv()
