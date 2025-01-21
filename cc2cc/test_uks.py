from timeit import default_timer as timer
import types

import numpy as np
import opt_einsum as oe

import pyscf
from pyscf import lib

from cc2cc.utils import DATA_PATH, AU2KCALMOL, GENERATE_DATA
from cc2cc.utils import Grid, TestData


def test_uks(
    mol,
    name,
    modeldict,
    data_record,
    lambda_=20,
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

    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_0 = ao_value[0]
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]

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

        if GENERATE_DATA:
            rho_a_diff = ni.eval_rho(mol, ao_0, dm[0] - test_data.dm1_cc[0])
            rho_b_diff = ni.eval_rho(mol, ao_0, dm[1] - test_data.dm1_cc[1])
            v_p_a = pyscf.dft.numint.eval_mat(
                mol, ao_0, grids.weights, rho_a_diff, rho_a_diff
            )
            v_p_b = pyscf.dft.numint.eval_mat(
                mol, ao_0, grids.weights, rho_b_diff, rho_b_diff
            )
            vxc += lambda_ * np.array([v_p_a, v_p_b])

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
    mdft.conv_tol = 1e-5
    mdft.conv_tol_grad = 1e-1

    mdft.kernel(dm0=test_data.mf_dm1)
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

    rho_scf = [
        pyscf.dft.numint.eval_rho(mol, ao_0, dm1_scf[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao_0, dm1_scf[1], xctype="LDA"),
    ]
    rho_cc = [
        pyscf.dft.numint.eval_rho(mol, ao_0, test_data.dm1_cc[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao_0, test_data.dm1_cc[1], xctype="LDA"),
    ]
    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao_0, test_data.dm1_dft[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao_0, test_data.dm1_dft[1], xctype="LDA"),
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

    if GENERATE_DATA:
        mf = pyscf.scf.UHF(mol)
        mf.kernel()
        mycc = pyscf.cc.UCCSD(mf)
        mycc.kernel()
        dm1_cc = np.array(mycc.make_rdm1(ao_repr=True))
        dm2_cc = np.array(mycc.make_rdm2(ao_repr=True))
        e_cc = mycc.e_tot
        dm1_dft = mdft.make_rdm1(ao_repr=True)

        test_dft = pyscf.scf.UKS(mol)
        test_dft.xc = "b3lyp"
        e_dft = test_dft.energy_tot(dm1_dft)

        rho_dft = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
        ]
        rho_cc = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA"),
        ]
        rho_cube = grids.gen_cube_rho(mol, dm1_dft)

        dm12 = (
            0.5 * dm2_cc[0]
            + 0.5 * dm2_cc[1]
            + 0.5 * dm2_cc[1].transpose(2, 3, 0, 1)
            + 0.5 * dm2_cc[2]
            - 0.5
            * oe.contract(
                "pq,rs->pqrs",
                dm1_dft[0] + dm1_dft[1],
                dm1_dft[0] + dm1_dft[1],
            )
            + 0.1
            * oe.contract(
                "pr,qs->pqrs",
                dm1_dft[0],
                dm1_dft[0],
            )
            + 0.1
            * oe.contract(
                "pr,qs->pqrs",
                dm1_dft[1],
                dm1_dft[1],
            )
        )

        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            dm12,
            (mol.nao,),
            (mol.nao,),
            (mol.nao, mol.nao),
            constants=[0],
            optimize="optimal",
        )

        exc_cc_grids = -pyscf.dft.libxc.eval_xc(
            "b3lyp", rho_dft, spin=1 if mol.spin else 0
        )[0] * (rho_dft[0][0] + rho_dft[1][0])

        for i, coord in enumerate(grids.coords):
            if i * 10 % len(grids.coords) == 0:
                print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)

            with mol.with_rinv_origin(coord):
                rinv = mol.intor("int1e_rinv")
                exc_cc_grids[i] += expr_rinv_dm2_r(
                    ao_value[0][i],
                    ao_value[0][i],
                    rinv,
                    backend="torch",
                )

        dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
        for i_spin in range(2):
            eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo[i_spin])
            eigs_v_dm1 = mf.mo_coeff[i_spin] @ eigs_v_dm1
            for i in range(np.shape(eigs_v_dm1)[1]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_value[0],
                    eigs_v_dm1[:, i],
                    eigs_v_dm1[:, i],
                    ao_2_diag,
                )
                exc_cc_grids -= part * eigs_e_dm1[i] / 2

            for i in range(mol.nelec[i_spin]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_value[0],
                    mdft.mo_coeff[i_spin][:, i],
                    mdft.mo_coeff[i_spin][:, i],
                    ao_2_diag,
                )
                exc_cc_grids += part / 2

        for i, coord in enumerate(grids.coords):
            for i_atom in range(mol.natm):
                distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
                if distance > 1e-3:
                    exc_cc_grids[i] -= (
                        (
                            rho_cc[0][0][i]
                            + rho_cc[1][0][i]
                            - rho_dft[0][0][i]
                            - rho_dft[1][0][i]
                        )
                        * mol.atom_charges()[i_atom]
                        / distance
                    )

        error_energy = e_cc - e_dft
        error = np.sum(exc_cc_grids * grids.weights) - error_energy
        print(
            "exc_cc_grids: ",
            f"max exc_cc_grids: {np.max(exc_cc_grids)}",
            f"min exc_cc_grids: {np.min(exc_cc_grids)}",
            f"mean exc_cc_grids: {np.mean(exc_cc_grids)}",
            f"std exc_cc_grids: {np.std(exc_cc_grids)}",
        )
        print(
            f"error_energy: {AU2KCALMOL * error_energy},",
            f"Error: {AU2KCALMOL * error},",
        )

        np.savez_compressed(
            DATA_PATH / f"data_{name}.npz",
            e_cc=e_cc,
            dm_cc=dm1_cc,
            rho_cube=rho_cube,
            weights=grids.weights,
            exc_cc_grids=exc_cc_grids,
            error_energy=error_energy,
        )

        print(f"Save data for {name}.")
