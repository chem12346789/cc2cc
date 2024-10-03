from pathlib import Path
import types

import pandas as pd
import numpy as np
import opt_einsum as oe

import pyscf
import pyscf.dft
from pyscf.grad import ccsd as ccsd_grad
from pyscf import lib

from dft2cc.utils import (
    MAIN_PATH,
    DATA_PATH,
    DATA_TEST_PATH,
    AU2KCALMOL,
    GENERATE_NEW,
    DATA_SCF_PATH,
)
from dft2cc.utils import gen_basis, rotate
from dft2cc.utils import Grid
from dft2cc.utils import process_input


class TEST_DATA:

    def __init__(
        self,
        molecular,
        name="methane",
        basis="sto-3g",
        if_basis_str=False,
        spin=0,
    ):
        self.name = name
        self.basis = basis
        self.if_basis_str = if_basis_str

        rotate(molecular)

        self.mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                self.basis,
                self.if_basis_str,
            ),
            verbose=4,
            spin=spin,
        )

    # pylint: disable=W0201
    def test_mol(self):
        """
        Generate 1-RDM.
        """
        # if False:
        if (DATA_TEST_PATH / f"data_{self.name}.npz").exists():
            print(f"Load data from {DATA_TEST_PATH}/data_{self.name}.npz")
            data_saved = np.load(f"{DATA_TEST_PATH}/data_{self.name}.npz")
            self.cc_dipole = data_saved["cc_dipole"]
            self.e_cc = data_saved["e_cc"]
            self.dm1_cc = data_saved["dm1_cc"]
            self.grad_ccsd = data_saved["grad_ccsd"]
            self.dm1_dft = data_saved["dm1_dft"]
            self.dft_dipole = data_saved["dft_dipole"]
            self.e_dft = data_saved["e_dft"]
            self.grad_dft = data_saved["grad_dft"]
        else:
            print(f"Generate data for {self.name}")

            mdft = pyscf.scf.RKS(self.mol)
            mdft.xc = "b3lyp"
            mdft.max_cycle = 250
            mdft.kernel()
            self.dm1_dft = mdft.make_rdm1(ao_repr=True)
            self.e_dft = mdft.e_tot
            g = mdft.nuc_grad_method()
            self.grad_dft = g.kernel()
            self.dft_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=self.dm1_dft,
                unit="A.U.",
            )

            mf = pyscf.scf.RHF(self.mol)
            mf.kernel()
            mycc = pyscf.cc.CCSD(mf)
            mycc.incore_complete = True
            mycc.async_io = False
            mycc.direct = True
            mycc.kernel()
            self.dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
            g = ccsd_grad.Gradients(mycc)
            self.grad_ccsd = g.kernel()
            self.cc_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=self.dm1_cc,
                unit="A.U.",
            )

            np.savez_compressed(
                Path(f"{MAIN_PATH}/data/test/data_{self.name}.npz"),
                cc_dipole=self.cc_dipole,
                e_cc=self.e_cc,
                dm1_cc=self.dm1_cc,
                grad_ccsd=self.grad_ccsd,
                dm1_dft=self.dm1_dft,
                dft_dipole=self.dft_dipole,
                e_dft=self.e_dft,
                grad_dft=self.grad_dft,
            )


def test_rks(
    args,
    molecular,
    name,
    modeldict,
    df_dict: dict,
    df_dict_path: Path,
):
    """
    Test the model. Restrict Khon-Sham (no spin).
    """
    # 2.0 Prepare
    test_data = TEST_DATA(
        molecular,
        name=name,
        basis=args.basis,
        if_basis_str=args.if_basis_str,
    )
    test_data.test_mol()

    grids = Grid(test_data.mol, level=1, period=2)
    ao_0 = pyscf.dft.numint.eval_ao(test_data.mol, grids.coords, deriv=0)

    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks.mol
        if dm is None:
            dm = ks.make_rdm1()

        ground_state = isinstance(dm, np.ndarray) and dm.ndim == 2
        ni = ks._numint

        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)

        correct_ene = modeldict.get_energy(ks, grids, dm)
        exc += correct_ene

        vxc_scf = modeldict.get_v(ks, grids, dm)
        vxc += pyscf.dft.numint.eval_mat(
            test_data.mol, ao_0, grids.weights, vxc_scf, vxc_scf
        )

        # rho_diff = ni.eval_rho(test_data.mol, ao_0, dm - test_data.dm1_cc)
        # v_p = pyscf.dft.numint.eval_mat(
        #     test_data.mol, ao_0, test_data.grids.weights, rho_diff, rho_diff
        # )
        # vxc += v_p

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

        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    mdft = pyscf.dft.RKS(test_data.mol)
    mdft.get_veff = types.MethodType(get_veff, mdft)
    mdft.xc = "b3lyp"
    if args.precision == "float32":
        mdft.conv_tol = 1e-4
    elif args.precision == "float64":
        mdft.conv_tol = 1e-8
    mdft.diis_space = 10
    mdft.DIIS = pyscf.scf.ADIIS
    mdft.max_cycle = 100
    mdft.level_shift = 0
    mdft.run()
    dm1_scf = mdft.make_rdm1()
    print("Done SCF", flush=True)

    rho_scf = mdft._numint.eval_rho(test_data.mol, ao_0, dm1_scf)
    rho_cc = mdft._numint.eval_rho(test_data.mol, ao_0, test_data.dm1_cc)
    rho_dft = mdft._numint.eval_rho(test_data.mol, ao_0, test_data.dm1_dft)
    print("Done rho_diff", flush=True)
    density_diff_scf = np.sum(np.abs(rho_scf - rho_cc) * grids.weights)
    density_diff_dft = np.sum(np.abs(rho_dft - rho_cc) * grids.weights)
    print("check", np.sum(rho_cc * grids.weights))

    df_dict["error_scf_ene"].append(AU2KCALMOL * (test_data.e_cc - mdft.e_tot))
    df_dict["error_dft_ene"].append(AU2KCALMOL * (test_data.e_cc - test_data.e_dft))
    df_dict["abs_cc_ene"].append(AU2KCALMOL * test_data.e_cc)

    df_dict["density_diff_scf"].append(density_diff_scf)
    df_dict["density_diff_dft"].append(density_diff_dft)
    df_dict["dipole_diff_scf"].append(0.0)
    df_dict["dipole_diff_dft"].append(0.0)
    df_dict["force_diff_scf"].append(0.0)
    df_dict["force_diff_dft"].append(0.0)

    # error_dipole = test_data.cc_dipole - test_data.dft_dipole
    # df_dict["dipole_diff_scf"].append(
    #     AU2DEBYE * np.linalg.norm(error_dipole - correct_dipole)
    # )
    # df_dict["dipole_diff_dft"].append(AU2DEBYE * np.linalg.norm(error_dipole))

    # error_force = test_data.grad_ccsd - test_data.grad_dft
    # df_dict["force_diff_scf"].append(
    #     AU2KCALMOL * np.linalg.norm(error_force - correct_force)
    # )
    # df_dict["force_diff_dft"].append(AU2KCALMOL * np.linalg.norm(error_force))

    df_dict["name"].append(name)
    print(df_dict)
    df = pd.DataFrame(df_dict)
    df.to_csv(df_dict_path, index=False)

    if GENERATE_NEW:
        if (DATA_PATH / f"data_{name}.npz").exists():
            data_load = np.load(DATA_PATH / f"data_{name}.npz")
            dm1_last = data_load["dm_cc"]
            exc_cc_grids = grids.matrix_to_vector(
                data_load["exc_over_dm_cc_grids_matrix"]
                * data_load["rho_inv_4_norm_matrix"][0, :, :, :]
            )
            rho_last = grids.matrix_to_vector(
                data_load["rho_inv_4_norm_matrix"][0, :, :, :]
            )

            ao_2 = pyscf.dft.numint.eval_ao(test_data.mol, grids.coords, deriv=2)
            ao_value = ao_2[:4, :, :]
            ao_2_diag = ao_2[4, :, :] + ao_2[7, :, :] + ao_2[9, :, :]

            exc_cc_grids += (
                pyscf.dft.libxc.eval_xc(
                    "b3lyp",
                    pyscf.dft.numint.eval_rho(
                        test_data.mol, ao_value, dm1_last, xctype="GGA"
                    ),
                )[0]
                * rho_last
                - pyscf.dft.libxc.eval_xc(
                    "b3lyp",
                    pyscf.dft.numint.eval_rho(
                        test_data.mol, ao_value, dm1_scf, xctype="GGA"
                    ),
                )[0]
                * rho_scf
            )

            expr_rinv_dm2_r = oe.contract_expression(
                "ijkl,i,j,kl->",
                -0.05 * oe.contract("pr,qs->pqrs", dm1_last, dm1_last)
                + 0.05 * oe.contract("pr,qs->pqrs", dm1_scf, dm1_scf),
                (test_data.mol.nao,),
                (test_data.mol.nao,),
                (test_data.mol.nao, test_data.mol.nao),
                constants=[0],
                optimize="optimal",
            )

            for i, coord in enumerate(grids.coords):
                if i * 10 % len(grids.coords) == 0:
                    print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)

                ao_0_i = ao_0[i]
                with test_data.mol.with_rinv_origin(coord):
                    rinv = test_data.mol.intor("int1e_rinv")
                    exc_cc_grids[i] += expr_rinv_dm2_r(
                        ao_0_i,
                        ao_0_i,
                        rinv,
                        backend="torch",
                    )

            exc_over_dm_cc_grids = exc_cc_grids / (rho_scf + 1e-14)
            print("Done exc_over_dm_cc_grids", flush=True)
            print("exc_cc_grids", np.max(exc_cc_grids))
            print("exc_cc_grids", np.min(exc_cc_grids))
            print("exc_over_dm_cc_grids", np.max(exc_over_dm_cc_grids))
            print("exc_over_dm_cc_grids", np.min(exc_over_dm_cc_grids))

            mat_s = mdft.get_ovlp()
            dm1_cc_mo = oe.contract(
                "ij,pi,qj->pq",
                test_data.dm1_cc,
                (mdft.mo_coeff).T @ mat_s,
                (mdft.mo_coeff).T @ mat_s,
            )

            eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_cc_mo)
            eigs_v_dm1 = mdft.mo_coeff @ eigs_v_dm1
            for i in range(np.shape(eigs_v_dm1)[1]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_0,
                    eigs_v_dm1[:, i],
                    eigs_v_dm1[:, i],
                    ao_2_diag,
                )
                exc_cc_grids -= part * eigs_e_dm1[i] / 2

            dm1_scf_mo = oe.contract(
                "ij,pi,qj->pq",
                dm1_scf,
                (mdft.mo_coeff).T @ mat_s,
                (mdft.mo_coeff).T @ mat_s,
            )
            eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(dm1_scf_mo)
            eigs_v_dm1 = mdft.mo_coeff @ eigs_v_dm1
            for i in range(np.shape(eigs_v_dm1)[1]):
                part = oe.contract(
                    "pm,m,n,pn->p",
                    ao_0,
                    eigs_v_dm1[:, i],
                    eigs_v_dm1[:, i],
                    ao_2_diag,
                )
                exc_cc_grids += part * eigs_e_dm1[i] / 2

            exc_over_dm_cc_grids = exc_cc_grids / (rho_scf + 1e-14)
            print("Done exc_over_dm_cc_grids", flush=True)
            print("exc_cc_grids", np.max(exc_cc_grids))
            print("exc_cc_grids", np.min(exc_cc_grids))
            print("exc_over_dm_cc_grids", np.max(exc_over_dm_cc_grids))
            print("exc_over_dm_cc_grids", np.min(exc_over_dm_cc_grids))

            np.savez_compressed(
                DATA_SCF_PATH / f"data_{name}.npz",
                dm1_cc=test_data.dm1_cc,
                rho_inv_4_norm_matrix=process_input(
                    pyscf.dft.numint.eval_rho(
                        test_data.mol, ao_value, dm1_scf, xctype="GGA"
                    ),
                    grids,
                ),
                exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(
                    exc_over_dm_cc_grids
                ),
                weights_matrix=grids.vector_to_matrix(grids.weights),
                error_energy=test_data.e_cc - mdft.e_tot,
            )
        else:
            print(f"Skip: {name:>40}")
