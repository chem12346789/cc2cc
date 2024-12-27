from timeit import default_timer as timer
import types

import numpy as np

import pyscf
from pyscf import lib

from cc2cc.utils import DATA_PATH, AU2KCALMOL
from cc2cc.utils import Grid, TestData


def test_uks(
    mol,
    name,
    modeldict,
    data_record,
):
    """
    Test the model. Restrict Khon-Sham (no spin).
    """
    # 2.0 Prepare
    test_data = TestData(mol, name, xc_code="b3lyp")
    test_data.test_mol()
    grids = Grid(test_data.mol)
    mdft = pyscf.dft.UKS(mol)
    mdft.xc = test_data.xc_code
    mdft.grids = grids

    time_ai_start = timer()

    def get_veff_modified(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks.mol

        if dm is None:
            dm = ks.make_rdm1()

        ground_state = dm.ndim == 3 and dm.shape[0] == 2
        ni = ks._numint

        nelec, exc, vxc = modeldict.get_nev(ni, ks, grids, dm, test_data.xc_code)

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
                vj = ks.get_j(mol, ddm[0] + ddm[1], hermi)
                vj += vhf_last.vj
            else:
                vj = ks.get_j(mol, dm[0] + dm[1], hermi)
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
                if omega != 0:
                    vklr = ks.get_k(mol, ddm, hermi, omega)
                    vklr *= alpha - hyb
                    vk += vklr
                vj = vj[0] + vj[1] + vhf_last.vj
                vk += vhf_last.vk
            else:
                vj, vk = ks.get_jk(mol, dm, hermi)
                vj = vj[0] + vj[1]
                vk *= hyb
                if omega != 0:
                    vklr = ks.get_k(mol, dm, hermi, omega)
                    vklr *= alpha - hyb
                    vk += vklr
            vxc += vj - vk

            if ground_state:
                exc -= (
                    np.einsum("ij,ji", dm[0], vk[0]).real
                    + np.einsum("ij,ji", dm[1], vk[1]).real
                ) * 0.5
        if ground_state:
            ecoul = np.einsum("ij,ji", dm[0] + dm[1], vj).real * 0.5
        else:
            ecoul = None

        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    mdft.get_veff = types.MethodType(get_veff_modified, mdft)
    mdft.conv_tol = 1e-6

    mdft.kernel()
    dm1_scf = mdft.make_rdm1()

    # mdft.max_cycle = -1
    # mdft.kernel(dm0=test_data.dm1_cc)
    # dm1_scf = test_data.dm1_dft.copy()

    scf_dipole = pyscf.scf.hf.dip_moment(
        mol=mol,
        dm=dm1_scf,
        unit="A.U.",
    )

    time_ai = timer() - time_ai_start

    error_dft_ene = AU2KCALMOL * (test_data.e_cc - test_data.e_dft)
    error_scf_ene = AU2KCALMOL * (test_data.e_cc - mdft.e_tot)
    error_dft_dip = np.linalg.norm(test_data.cc_dipole - test_data.dft_dipole)
    error_scf_dip = np.linalg.norm(test_data.cc_dipole - scf_dipole)

    grids = Grid(mol)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=0)
    rho_scf = pyscf.dft.numint.eval_rho(
        mol, ao_value, dm1_scf[0] + dm1_scf[1], xctype="LDA"
    )
    rho_cc = pyscf.dft.numint.eval_rho(
        mol, ao_value, test_data.dm1_cc[0] + test_data.dm1_cc[1], xctype="LDA"
    )
    rho_dft = pyscf.dft.numint.eval_rho(
        mol, ao_value, test_data.dm1_dft[0] + test_data.dm1_dft[1], xctype="LDA"
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
