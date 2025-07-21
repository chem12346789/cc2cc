from timeit import default_timer as timer
import os
import numpy as np

import pyscf
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad import uccsd as uccsd_grad

from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import ccsd_t_slow as ccsd_t

from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
from pyscf.cc import uccsd_t

from cc2cc.utils.env_var import DATA_TEST_PATH, DATA_TEST_NO_GRAD_PATH


class TestData:
    """
    Class to generate and store test data for CCSD and DFT calculations.
    It generates 1-RDM, energy, dipole, and gradient for a given molecule.
    The data is saved in a compressed npz file for later use.
    Note:
        1) If the data already exists, it will be loaded instead of recomputed.
        2) If the molecule coordinates are different from the saved data, it will raise an error.
        3) If disp is not None, it will generate data for the dispersion-corrected DFT calculation (Will store the data in the same file).
    Args:
        mol (pyscf.Mole): The molecule object.
        name (str): The name of the molecule, used for saving/loading data.
        xc_code (str): The exchange-correlation functional code for DFT calculations.
        if_grad (bool): If True, compute gradients.
            TODO: This function is under development.
        cc_triple (bool): If True, include CCSD(T) corrections.
        disp (str or None): Dispersion correction method, if any. Default is None.
    Raises:
        ValueError: If the molecule coordinates are different from the saved data.
    """

    def __init__(
        self,
        mol: pyscf.M,
        name: str,
        xc_code: str = "b3lyp",
        if_grad: bool = False,
        cc_triple: bool = False,
        if_disp: str = None,
    ) -> None:
        self.name = name
        self.mol = mol
        self.xc_code = xc_code

        if if_grad:
            path_to_data = DATA_TEST_PATH
        else:
            path_to_data = DATA_TEST_NO_GRAD_PATH

        if (path_to_data / f"{name}_cc.npz").exists():
            data_frame = dict(
                np.load(path_to_data / f"{name}_cc.npz", allow_pickle=True).items()
            )
        else:
            self.mf_dm1 = None
            data_frame = {"mol_corr": mol.atom_coords(), "mf_dm1": self.mf_dm1}

            if mol.spin == 0:
                self.test_mol_rcc
                data_frame_cc = self.test_mol_rcc(if_grad=if_grad, cc_triple=cc_triple)
                data_frame_ks = self.test_mol_rks(if_grad=if_grad, disp=None)
            else:
                data_frame_cc = self.test_mol_ucc(if_grad=if_grad, cc_triple=cc_triple)
                data_frame_ks = self.test_mol_uks(if_grad=if_grad, disp=None)
            data_frame.update(data_frame_cc)
            data_frame.update(data_frame_ks)

            np.savez_compressed(path_to_data / f"{name}_cc.npz", **data_frame)

        print(f"Data for {name} loaded from file.")
        mol_corr = data_frame["mol_corr"]

        if np.linalg.norm(mol.atom_coords() - mol_corr, ord=1) > 1e-6:
            print("Molecule coordinates are different.")
            print("With nothing to do, skip the test.")
            raise ValueError("Molecule coordinates are different.")

        self.mf_dm1 = data_frame["mf_dm1"]

        self.dm1_cc = data_frame["dm1_cc"]
        self.e_cc = data_frame["e_cc"].item()
        self.cc_dipole = data_frame["cc_dipole"]
        self.time_cc = data_frame["time_cc"].item()
        self.grad_ccsd = data_frame["grad_ccsd"]

        self.dm1_dft = data_frame["dm1_dft"]
        self.e_dft = data_frame["e_dft"].item()
        self.dft_dipole = data_frame["dft_dipole"]
        self.time_dft = data_frame["time_dft"].item()
        self.grad_dft = data_frame["grad_dft"]

        if if_disp:
            self.delta_e = {}
            for disp in ["d3zero", "d3bj"]:
                if not (
                    f"dm1_dft_{disp}" in data_frame
                    and f"e_dft_{disp}" in data_frame
                    and f"dft_dipole_{disp}" in data_frame
                    and f"time_dft_{disp}" in data_frame
                    and f"grad_dft_{disp}" in data_frame
                ):
                    # if True:
                    print(f"Dispersion {disp} not found in data, generating...")
                    if mol.spin == 0:
                        data_frame_ks = self.test_mol_rks(if_grad=if_grad, disp=disp)
                    else:
                        data_frame_ks = self.test_mol_uks(if_grad=if_grad, disp=disp)

                    data_frame.update(data_frame_ks)
                    np.savez_compressed(
                        path_to_data / f"{name}_cc.npz",
                        **data_frame,
                    )

                self.delta_e[disp] = (
                    data_frame[f"e_dft_{disp}"].item() - data_frame["e_dft"].item()
                )

        print(f"Data for {name} loaded.")
        print(f"CCSD energy: {self.e_cc}")
        print(f"DFT energy: {self.e_dft}")

    def test_mol_rcc(self, if_grad=False, cc_triple=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        time_start = timer()
        mf = pyscf.scf.RHF(self.mol)
        mf.max_cycle = 200
        mf.diis_space = 12
        mf.verbose = 4

        if "C60ISO" in self.name or "UPU23" in self.name:
            mf = mf.density_fit().run()
            self.mf_dm1 = mf.make_rdm1()
            mycc = pyscf.cc.CCSD(mf)
            mycc.max_cycle = 200
            # mycc.direct = True # This is not working for density_fit
            mycc.set_frozen()
            print(f"Number of core orbital frozen: {mycc.frozen}")
        else:
            mf.kernel()
            self.mf_dm1 = mf.make_rdm1()
            mycc = pyscf.cc.CCSD(mf)
            # mycc.direct = True
            mycc.max_cycle = 200

        mycc.verbose = 4
        _, t1, t2 = mycc.kernel()
        if mycc.converged is False:
            raise ValueError("CCSD not converged.")
        if cc_triple:
            eris = mycc.ao2mo()
            e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
            l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
            dm1_cc = ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris, ao_repr=True)
            e_cc = mycc.e_tot + e3ref
        else:
            dm1_cc = mycc.make_rdm1(ao_repr=True)
            e_cc = mycc.e_tot
        dm1_cc = np.array(dm1_cc)
        cc_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=dm1_cc,
            unit="A.U.",
        )
        if if_grad:
            g = ccsd_grad.Gradients(mycc)
            grad_ccsd = g.kernel()
        else:
            grad_ccsd = None
        time_cc = timer() - time_start
        return {
            "dm1_cc": dm1_cc,
            "e_cc": e_cc,
            "cc_dipole": cc_dipole,
            "time_cc": time_cc,
            "grad_ccsd": grad_ccsd,
        }

    def test_mol_ucc(self, if_grad=False, cc_triple=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        time_start = timer()
        mf = pyscf.scf.UHF(self.mol)
        mf.max_cycle = 200
        mf.diis_space = 12
        mf.verbose = 4

        if "C60ISO" in self.name or "UPU23" in self.name:
            mf = mf.density_fit().run()
            self.mf_dm1 = mf.make_rdm1()
            mycc = pyscf.cc.UCCSD(mf)
            mycc.max_cycle = 200
            # mycc.direct = True # This is not working for density_fit
            mycc.set_frozen()
            print(f"Number of core orbital frozen: {mycc.frozen}")
        else:
            mf.kernel()
            self.mf_dm1 = mf.make_rdm1()
            mycc = pyscf.cc.UCCSD(mf)
            # mycc.direct = True
            mycc.max_cycle = 200

        mycc.verbose = 4
        _, t1, t2 = mycc.kernel()
        if mycc.converged is False:
            raise ValueError("UCCSD not converged.")
        if cc_triple:
            eris = mycc.ao2mo()
            e3ref = uccsd_t.kernel(mycc, eris, t1, t2)
            l1, l2 = uccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
            dm1_cc = uccsd_t_rdm.make_rdm1(
                mycc, t1, t2, l1, l2, eris=eris, ao_repr=True
            )
            e_cc = mycc.e_tot + e3ref
        else:
            dm1_cc = mycc.make_rdm1(ao_repr=True)
            e_cc = mycc.e_tot
        dm1_cc = np.array(dm1_cc)
        cc_dipole = pyscf.scf.uhf.dip_moment(
            mol=self.mol,
            dm=dm1_cc,
            unit="A.U.",
        )
        if if_grad:
            g = uccsd_grad.Gradients(mycc)
            grad_ccsd = g.kernel()
        else:
            grad_ccsd = None
        time_cc = timer() - time_start
        return {
            "dm1_cc": dm1_cc,
            "e_cc": e_cc,
            "cc_dipole": cc_dipole,
            "time_cc": time_cc,
            "grad_ccsd": grad_ccsd,
        }

    def test_mol_rks(self, if_grad=False, disp=None):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected RKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.RKS(self.mol)
        if disp is not None:
            mdft.xc = f"{self.xc_code}-{disp}"
            name_disp = f"_{disp}"
        else:
            mdft.xc = self.xc_code
            name_disp = ""
        mdft.max_cycle = 250
        mdft.verbose = 4
        mdft.kernel(dm0=self.mf_dm1)
        if mdft.converged is False:
            raise ValueError("RKS not converged.")
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        e_dft = mdft.e_tot
        dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=dm1_dft,
            unit="A.U.",
        )
        if if_grad:
            g = mdft.nuc_grad_method()
            grad_dft = g.kernel()
        else:
            grad_dft = None
        time_dft = timer() - time_start
        return {
            f"dm1_dft{name_disp}": dm1_dft,
            f"e_dft{name_disp}": e_dft,
            f"dft_dipole{name_disp}": dft_dipole,
            f"time_dft{name_disp}": time_dft,
            f"grad_dft{name_disp}": grad_dft,
        }

    def test_mol_uks(self, if_grad=False, disp=None):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected UKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.UKS(self.mol)
        if disp is not None:
            mdft.xc = f"{self.xc_code}-{disp}"
            name_disp = f"_{disp}"
        else:
            mdft.xc = self.xc_code
            name_disp = ""
        mdft.max_cycle = 250
        mdft.verbose = 4
        mdft.kernel(dm0=self.mf_dm1)
        if mdft.converged is False:
            raise ValueError("UKS not converged.")
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        e_dft = mdft.e_tot
        dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=dm1_dft,
            unit="A.U.",
        )
        if if_grad:
            g = mdft.nuc_grad_method()
            grad_dft = g.kernel()
        else:
            grad_dft = None
        time_dft = timer() - time_start
        return {
            f"dm1_dft{name_disp}": dm1_dft,
            f"e_dft{name_disp}": e_dft,
            f"dft_dipole{name_disp}": dft_dipole,
            f"time_dft{name_disp}": time_dft,
            f"grad_dft{name_disp}": grad_dft,
        }

    def test_mol_orca(self, if_grad=False, cc_triple=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        molecular_xyz = ""
        for atom_info in self.mol._atom:
            molecular_xyz += (
                f"{atom_info[0]:<6}\t{atom_info[1][0]:<16.10}\t{atom_info[1][1]:<16.10}\t{atom_info[1][2]:<16.10}"
                + "\n"
            )

        with open(f"tmp_mol/{self.name}.inp", "w", encoding="utf-8") as f:
            f.write(
                f"""! cc-pVDZ cc-pVDZ/C DLPNO-CCSD TightSCF
        %method
          WriteJSONPropertyfile True
        end

        %MDCI 
          Density Unrelaxed
        end

        %pal
          nprocs {os.environ.get("OMP_NUM_THREADS")}
        end

        %maxcore {os.environ.get("PYSCF_MAX_MEMORY")}

        %coords
        CTyp   xyz     # the type of coordinates = xyz or internal
        Charge {self.mol.charge}       # the total charge of the molecule
        Mult   {self.mol.spin+1}        # the multiplicity = 2S+1
        Units  bohrs    # the unit of length = angs or bohrs

        # the subblock coords is for the actual coordinates
        # for CTyp=xyz
        coords
        {molecular_xyz}end
        end
        """
            )

        os.system(f"$(which orca) tmp_mol/{self.name}.inp > tmp_mol/{self.name}.out")
