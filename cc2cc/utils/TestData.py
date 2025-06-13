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

from cc2cc.utils.env_var import DATA_TEST_PATH


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
        disp: str = None,
    ) -> None:
        self.name = name
        self.mol = mol
        self.xc_code = xc_code
        self.disp = disp

        if (DATA_TEST_PATH / f"{name}_cc.npz").exists():
            data_frame = dict(
                np.load(DATA_TEST_PATH / f"{name}_cc.npz", allow_pickle=True).items()
            )
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
            if if_grad:
                if "grad_ccsd" not in data_frame:
                    raise ValueError("No gradient data.")
                self.grad_ccsd = data_frame["grad_ccsd"]

            if disp is None:
                self.dm1_dft = data_frame["dm1_dft"]
                self.e_dft = data_frame["e_dft"].item()
                self.dft_dipole = data_frame["dft_dipole"]
                self.time_dft = data_frame["time_dft"].item()
                if if_grad:
                    if "grad_dft" not in data_frame:
                        raise ValueError("No DFT gradient data.")
                    self.grad_dft = data_frame["grad_dft"]
            else:
                if (
                    f"dm1_dft_{disp}" in data_frame
                    and f"e_dft_{disp}" in data_frame
                    and f"dft_dipole_{disp}" in data_frame
                    and f"time_dft_{disp}" in data_frame
                ):
                    self.dm1_dft = data_frame[f"dm1_dft_{disp}"]
                    self.e_dft = data_frame[f"e_dft_{disp}"].item()
                    self.dft_dipole = data_frame[f"dft_dipole_{disp}"]
                    self.time_dft = data_frame[f"time_dft_{disp}"].item()
                    if if_grad:
                        if "grad_dft" not in data_frame:
                            raise ValueError("No DFT gradient data.")
                        self.grad_dft = data_frame[f"grad_dft_{disp}"]
                else:
                    self.dm1_dft = None
                    self.e_dft = None
                    self.dft_dipole = None
                    self.time_dft = None
                    if mol.spin == 0:
                        self.test_mol_rks_disp(if_grad)
                    else:
                        self.test_mol_uks_disp(if_grad)

                    data_frame.update(
                        {
                            f"dm1_dft_{disp}": self.dm1_dft,
                            f"e_dft_{disp}": self.e_dft,
                            f"dft_dipole_{disp}": self.dft_dipole,
                            f"time_dft_{disp}": self.time_dft,
                        }
                    )
                    np.savez_compressed(
                        DATA_TEST_PATH / f"{name}_cc.npz",
                        **data_frame,
                    )

            print(f"Data for {name} loaded.")
            print(f"CCSD energy: {self.e_cc}")
            print(f"DFT energy: {self.e_dft}")
        else:
            self.mf_dm1 = None
            self.dm1_cc = None
            self.e_cc = None
            self.cc_dipole = None
            if mol.spin == 0:
                self.test_mol_rks(if_grad=if_grad, cc_triple=cc_triple)
            else:
                self.test_mol_uks(if_grad=if_grad, cc_triple=cc_triple)

            np.savez_compressed(
                DATA_TEST_PATH / f"{name}_cc.npz",
                mol_corr=mol.atom_coords(),
                mf_dm1=self.mf_dm1,
                dm1_cc=self.dm1_cc,
                e_cc=self.e_cc,
                cc_dipole=self.cc_dipole,
                time_cc=self.time_cc,
                grad_ccsd=self.grad_ccsd if if_grad else None,
                dm1_dft=self.dm1_dft,
                e_dft=self.e_dft,
                dft_dipole=self.dft_dipole,
                time_dft=self.time_dft,
                grad_dft=self.grad_dft if if_grad else None,
            )

    def test_mol_rks(self, if_grad=False, cc_triple=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        time_start = timer()
        mf = pyscf.scf.RHF(self.mol)
        mf.max_cycle = 200
        mf.diis_space = 12

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
            mycc.direct = True
            mycc.max_cycle = 200

        _, t1, t2 = mycc.kernel()
        if mycc.converged is False:
            raise ValueError("CCSD not converged.")
        if cc_triple:
            eris = mycc.ao2mo()
            e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
            l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
            self.dm1_cc = ccsd_t_rdm.make_rdm1(
                mycc, t1, t2, l1, l2, eris=eris, ao_repr=True
            )
            self.e_cc = mycc.e_tot + e3ref
        else:
            self.dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
        self.dm1_cc = np.array(self.dm1_cc)
        self.cc_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=self.dm1_cc,
            unit="A.U.",
        )
        if if_grad:
            g = ccsd_grad.Gradients(mycc)
            self.grad_ccsd = g.kernel()
        self.time_cc = timer() - time_start

        time_start = timer()
        mdft = pyscf.scf.RKS(self.mol)
        mdft.xc = self.xc_code
        mdft.max_cycle = 250
        mdft.kernel(dm0=self.mf_dm1)
        if mdft.converged is False:
            raise ValueError("RKS not converged.")
        self.dm1_dft = mdft.make_rdm1(ao_repr=True)
        self.e_dft = mdft.e_tot
        self.dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=self.dm1_dft,
            unit="A.U.",
        )
        if if_grad:
            g = mdft.nuc_grad_method()
            self.grad_dft = g.kernel()
        self.time_dft = timer() - time_start

    def test_mol_uks(self, if_grad=False, cc_triple=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        time_start = timer()
        mf = pyscf.scf.UHF(self.mol)
        mf.max_cycle = 200
        mf.diis_space = 12

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
            mycc.direct = True
            mycc.max_cycle = 200

        _, t1, t2 = mycc.kernel()
        if mycc.converged is False:
            raise ValueError("UCCSD not converged.")
        if cc_triple:
            eris = mycc.ao2mo()
            e3ref = uccsd_t.kernel(mycc, eris, t1, t2)
            l1, l2 = uccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
            self.dm1_cc = uccsd_t_rdm.make_rdm1(
                mycc, t1, t2, l1, l2, eris=eris, ao_repr=True
            )
            self.e_cc = mycc.e_tot + e3ref
        else:
            self.dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
        self.dm1_cc = np.array(self.dm1_cc)
        self.cc_dipole = pyscf.scf.uhf.dip_moment(
            mol=self.mol,
            dm=self.dm1_cc,
            unit="A.U.",
        )
        if if_grad:
            g = uccsd_grad.Gradients(mycc)
            self.grad_ccsd = g.kernel()
        self.time_cc = timer() - time_start

        time_start = timer()
        mdft = pyscf.scf.UKS(self.mol)
        mdft.xc = self.xc_code
        mdft.max_cycle = 250
        mdft.kernel(dm0=self.mf_dm1)
        if mdft.converged is False:
            raise ValueError("UKS not converged.")
        self.dm1_dft = mdft.make_rdm1(ao_repr=True)
        self.e_dft = mdft.e_tot
        self.dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=self.dm1_dft,
            unit="A.U.",
        )
        if if_grad:
            g = mdft.nuc_grad_method()
            self.grad_dft = g.kernel()
        self.time_dft = timer() - time_start

    def test_mol_rks_disp(self, if_grad=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected RKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.RKS(self.mol)
        mdft.xc = self.xc_code
        mdft.disp = self.disp
        mdft.max_cycle = 250
        mdft.kernel(dm0=self.mf_dm1)
        if mdft.converged is False:
            raise ValueError("RKS not converged.")
        self.dm1_dft = mdft.make_rdm1(ao_repr=True)
        self.e_dft = mdft.e_tot
        self.dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=self.dm1_dft,
            unit="A.U.",
        )
        if if_grad:
            g = mdft.nuc_grad_method()
            self.grad_dft = g.kernel()
        self.time_dft = timer() - time_start

    def test_mol_uks_disp(self, if_grad=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected UKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.UKS(self.mol)
        mdft.xc = self.xc_code
        mdft.disp = self.disp
        mdft.max_cycle = 250
        mdft.kernel(dm0=self.mf_dm1)
        if mdft.converged is False:
            raise ValueError("UKS not converged.")
        self.dm1_dft = mdft.make_rdm1(ao_repr=True)
        self.e_dft = mdft.e_tot
        self.dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=self.dm1_dft,
            unit="A.U.",
        )
        if if_grad:
            g = mdft.nuc_grad_method()
            self.grad_dft = g.kernel()
        self.time_dft = timer() - time_start

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
                f"""! DLPNO-CCSD
        %basis
        # read an externally specified orbital basis
        GTOName      = "cc-pvdz.1.orca"
        Aux "AutoAux"
        AuxJK "AutoAux"
        AuxC "AutoAux"
        end
        %method
        WriteJSONPropertyfile True
        end
        %MDCI Density Unrelaxed
        end
        %pal nprocs {os.environ.get("OMP_NUM_THREADS")} end
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
