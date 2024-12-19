from timeit import default_timer as timer

import pyscf
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad import uccsd as uccsd_grad


class TestData:

    def __init__(
        self,
        mol,
        name="methane",
        xc_code="b3lyp",
    ):
        self.name = name
        self.mol = mol
        self.xc_code = xc_code

    # pylint: disable=W0201
    def test_mol(self, if_grad=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        if self.mol.spin == 0:
            time_start = timer()
            mdft = pyscf.scf.RKS(self.mol)
            mdft.xc = self.xc_code
            mdft.max_cycle = 250
            mdft.kernel()
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
            self.cc_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=self.dm1_cc,
                unit="A.U.",
            )
            if if_grad:
                g = ccsd_grad.Gradients(mycc)
                self.grad_ccsd = g.kernel()
            self.time_cc = timer() - time_start
        else:
            time_start = timer()
            mdft = pyscf.scf.UKS(self.mol)
            mdft.xc = self.xc_code
            mdft.max_cycle = 250
            mdft.kernel()
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
            self.cc_dipole = pyscf.scf.uhf.dip_moment(
                mol=self.mol,
                dm=self.dm1_cc,
                unit="A.U.",
            )
            if if_grad:
                g = uccsd_grad.Gradients(mycc)
                self.grad_ccsd = g.kernel()
            self.time_cc = timer() - time_start
