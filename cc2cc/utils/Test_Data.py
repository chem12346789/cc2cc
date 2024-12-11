import numpy as np
from timeit import default_timer as timer

import pyscf
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad import uccsd as uccsd_grad

from cc2cc.utils import DATA_TEST_PATH


class Test_Data:

    def __init__(
        self,
        mol,
        name="methane",
    ):
        self.name = name
        self.mol = mol

    # pylint: disable=W0201
    def test_mol(self):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        if self.mol.spin == 0:
            time_start = timer()
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
            self.time_dft = timer() - time_start

            time_start = timer()
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
            self.time_cc = timer() - time_start
        else:
            time_start = timer()
            mdft = pyscf.scf.UKS(self.mol)
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
            self.time_dft = timer() - time_start

            time_start = timer()
            mf = pyscf.scf.UHF(self.mol)
            mf.kernel()
            mycc = pyscf.cc.UCCSD(mf)
            mycc.incore_complete = True
            mycc.async_io = False
            mycc.direct = True
            mycc.kernel()
            self.dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
            g = uccsd_grad.Gradients(mycc)
            self.grad_ccsd = g.kernel()
            self.cc_dipole = pyscf.scf.uhf.dip_moment(
                mol=self.mol,
                dm=self.dm1_cc,
                unit="A.U.",
            )
            self.time_cc = timer() - time_start

        np.savez_compressed(
            DATA_TEST_PATH / f"data_{self.name}.npz",
            cc_dipole=self.cc_dipole,
            e_cc=self.e_cc,
            dm1_cc=self.dm1_cc,
            grad_ccsd=self.grad_ccsd,
            time_cc=self.time_cc,
            e_dft=self.e_dft,
            dm1_dft=self.dm1_dft,
            dft_dipole=self.dft_dipole,
            grad_dft=self.grad_dft,
            time_dft=self.time_dft,
        )
