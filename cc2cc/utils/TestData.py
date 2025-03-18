from timeit import default_timer as timer

import numpy as np

import pyscf
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad import uccsd as uccsd_grad

from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
from pyscf.cc import ccsd_t_slow as ccsd_t
from pyscf.cc import ccsd_rdm
from pyscf.cc.ccsd_t_rdm_slow import _gamma1_intermediates
from pyscf.cc.ccsd_t_rdm_slow import _gamma2_intermediates

from pyscf.cc import uccsd_t_lambda as uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm as uccsd_t_rdm
from pyscf.cc import uccsd_t as uccsd_t
from pyscf.cc import uccsd_rdm
from pyscf.cc.uccsd_t_rdm import _gamma1_intermediates as u_gamma1_intermediates
from pyscf.cc.uccsd_t_rdm import _gamma2_intermediates as u_gamma2_intermediates


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
    def test_mol_rks(self, if_grad=False, cc_triple=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        time_start = timer()
        mf = pyscf.scf.RHF(self.mol)
        mf.kernel()
        if mf.converged is False:
            raise ValueError("RHF not converged.")
        self.mf_dm1 = mf.make_rdm1()
        mycc = pyscf.cc.CCSD(mf)
        _, t1, t2 = mycc.kernel()
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

    # pylint: disable=W0201
    def test_mol_uks(self, if_grad=False, cc_triple=False):
        """
        Generate 1-RDM, energy, dipole, and gradient for the molecule.
        """
        print(f"Generate data for {self.name}")

        time_start = timer()
        mf = pyscf.scf.UHF(self.mol)
        mf.kernel()
        if mf.converged is False:
            raise ValueError("UHF not converged.")
        self.mf_dm1 = mf.make_rdm1()
        mycc = pyscf.cc.UCCSD(mf)
        _, t1, t2 = mycc.kernel()
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
