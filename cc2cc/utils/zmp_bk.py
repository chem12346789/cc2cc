"""
ZMP is a method to invert the density matrix from CCSD(T). See the original paper for details: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.50.2138 and https://arxiv.org/pdf/2603.22140v1.
It is based on the idea of minimizing the difference between the DFT and CC density matrices in a least-squares sense.
"""

import types

import numpy as np
from scipy.linalg import fractional_matrix_power

import pyscf
from pyscf import lib
from pyscf.lib import logger
import pyscf.dft


class RZMP(pyscf.dft.rks.RKS):

    def __init__(self, mol, dm_tar, dm_guide=None, faxc=0, dftxc=0):
        super().__init__(mol)
        self.initialize_grids(mol, dm_tar)

        self.faxc = faxc
        self.dftxc = dftxc
        self.dm_tar = dm_tar
        self.dm = dm_tar
        self.J_tar = self.get_j(self.mol, self.dm_tar)
        s1e = mol.intor_symmetric("int1e_ovlp")
        self.s_half = fractional_matrix_power(s1e, 0.5)

        if dm_guide is None:
            self.dm_guide = self.dm_tar
        else:
            self.dm_guide = dm_guide

        J_guide = self.get_j(self.mol, self.dm_guide)
        self.extra_vxc = -(self.faxc / self.mol.nelectron) * J_guide
        ni = self._numint
        vxc = ni.nr_rks(mol, self.grids, self.xc, self.dm_guide)[2]
        self.extra_vxc += self.dftxc * vxc

    def get_veff(
        self,
        mol=None,
        dm=None,
        dm_last=0,
        vhf_last=0,
        hermi=1,
    ):
        """
        # Get the effective potential for the RKS method.
        # This function is used to get the effective potential for the RKS method.
        # Modified from pyscf.dft.rks.get_veff; See
        # https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/dft/rks.py
        """
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = self.mol

        if dm is None:
            dm = self.make_rdm1()

        # self.initialize_grids(mol, dm)
        t0 = (logger.process_clock(), logger.perf_counter())

        vj = self.get_j(mol, dm, hermi)
        vxc = self.extra_vxc
        t0 = logger.timer(self, "jk", *t0)

        if self.if_ZMP:
            # print("Using ZMP potential", flush=True)
            vxc += self.lambda_ZMP * (vj - self.J_tar)

        if self.if_DMP:
            # print("Using DMP potential", flush=True)
            vxc += (
                2
                * self.lambda_DMP
                * self.s_half
                @ np.transpose(dm - self.dm_tar)
                @ self.s_half
            )

        t0 = logger.timer(self, "MP", *t0)

        ground_state = isinstance(dm, np.ndarray) and dm.ndim == 2
        if ground_state:
            ecoul = np.einsum("ij,ji", dm, vj).real * 0.5
        else:
            ecoul = None

        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=0, vj=vj, vk=None)

        return vxc

    def zscf(self, lambda_ZMP=None, lambda_DMP=None):
        if lambda_ZMP is not None:
            self.lambda_ZMP = lambda_ZMP
            self.if_ZMP = True
        else:
            self.if_ZMP = False

        if lambda_DMP is not None:
            self.lambda_DMP = lambda_DMP
            self.if_DMP = True
        else:
            self.if_DMP = False

        self.kernel(self.dm)
        self.dm = self.make_rdm1()
        print(
            f"Use cycles = {self.cycles}, and final convergence = {self.converged}",
            flush=True,
        )


def uzmp(dm1_dft, dm1_cc):
    return
