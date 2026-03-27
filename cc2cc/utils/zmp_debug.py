"""
ZMP is a method to invert the density matrix from CCSD(T). See the original paper for details: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.50.2138 and https://arxiv.org/pdf/2603.22140v1.
It is based on the idea of minimizing the difference between the DFT and CC density matrices in a least-squares sense.
"""

import types
from functools import reduce

import numpy as np
from scipy.linalg import fractional_matrix_power

import pyscf
from pyscf import lib
from pyscf.lib import logger
import pyscf.dft


class DIIS:
    """Summary: Class for DIIS extrapolation used in ZMP

    See [Pulay1980]_ for some extra context.

    """

    def __init__(self, S, diis_space):
        """Initialize DIIS object

        Args:
            S (ndarray):  overlap integral
            diis_space (integer) : number of DIIS vectors used in extrapolation

        """
        eig, Z = np.linalg.eigh(S)
        S12 = 1.0 / np.sqrt(eig)
        self.S = S
        self.O = reduce(np.dot, (Z, np.diag(S12), Z.T))
        self.diis_space = diis_space
        self.norb = len(S[0])
        self.ems = np.zeros((self.diis_space, self.norb, self.norb))
        self.pms = np.zeros((self.diis_space, self.norb, self.norb))
        self.tall = self.t_1 = self.t_2 = self.t_3 = 0.0

    def extrapolate(self, iteration, fock, dm):
        """Summary: New fock matrix by linear combination of previous fock matrices

        Args:
            iteration (integer) : present SCF iteration
            fock (ndarray) : fock matrix
            dm (ndarray) : density matrix obtained from previous step

        Returns:
            (tuple): tuple containing:

                (ndarray): **newfock** extrapolated fock matrix

                (float): **diis_error** DIIS error used in convergence criteria

        """

        if iteration <= 1 or self.diis_space < 2:
            return fock, 0.0

        for k in range(1, min(iteration, self.diis_space))[::-1]:
            self.ems[k] = self.ems[k - 1]
            self.pms[k] = self.pms[k - 1]

        em = reduce(np.dot, (fock, dm, self.S))
        em -= em.T
        self.ems[0] = reduce(np.dot, (self.O.T, em, self.O))
        self.pms[0] = fock[:]
        idx = np.abs(self.ems[0]).argmax()
        diis_error = np.abs(np.ravel(self.ems[0])[idx])

        # Solve BC = A to find C
        nb = min(iteration, self.diis_space) - 1
        B = -1.0 * np.ones((nb + 1, nb + 1))
        B[nb, nb] = 0.0
        B[:nb, :nb] = np.einsum(
            "aij,bji->ab", self.ems[:nb, :, :], self.ems[:nb, :, :], optimize="greedy"
        )
        A = np.zeros(nb + 1)
        A[nb] = -1.0
        C = np.linalg.solve(B, A)

        # form new extrapolated diis fock matrix
        newfock = np.zeros_like(fock)
        for i, c in enumerate(C[:-1]):
            newfock += c * self.pms[i]

        return newfock, diis_error


class RZMP(pyscf.dft.rks.RKS):

    def __init__(self, mol, dm_tar, dm_guide=None, faxc=0, dftxc=0):
        super().__init__(mol)
        self.initialize_grids(mol, dm_tar)

        self.faxc = faxc
        self.dftxc = dftxc
        self.dm_tar = dm_tar
        self.dm = dm_tar
        self.dm_old = dm_tar
        self.J_tar = self.get_j(self.mol, self.dm_tar)
        self.S = mol.intor_symmetric("int1e_ovlp")
        self.s_half = fractional_matrix_power(self.S, 0.5)
        self.DIIS = pyscf.scf.diis.EDIIS

        self.conv_tol_dm = 1e-7
        self.conv_tol_diis = 1e-5

        if dm_guide is None:
            self.dm_guide = self.dm_tar
        else:
            self.dm_guide = dm_guide

        J_guide = self.get_j(self.mol, self.dm_guide)
        self.extra_vxc = -(self.faxc / self.mol.nelectron) * J_guide
        ni = self._numint
        vxc = ni.nr_rks(self.mol, self.grids, self.xc, self.dm_guide)[2]
        self.extra_vxc += self.dftxc * vxc
        self.F0 = (
            self.mol.intor_symmetric("int1e_kin")
            + self.mol.intor_symmetric("int1e_nuc")
            + self.extra_vxc
        )

    # def get_veff(
    #     self,
    #     mol=None,
    #     dm=None,
    #     dm_last=0,
    #     vhf_last=0,
    #     hermi=1,
    # ):
    #     """
    #     # Get the effective potential for the RKS method.
    #     # This function is used to get the effective potential for the RKS method.
    #     # Modified from pyscf.dft.rks.get_veff; See
    #     # https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/dft/rks.py
    #     """
    #     # print("Using modified get_veff", flush=True)
    #     if mol is None:
    #         mol = self.mol

    #     if dm is None:
    #         dm = self.make_rdm1()

    #     # self.initialize_grids(mol, dm)
    #     t0 = (logger.process_clock(), logger.perf_counter())

    #     ground_state = isinstance(dm, np.ndarray) and dm.ndim == 2

    #     incremental_jk = (
    #         self._eri is None
    #         and self.direct_scf
    #         and getattr(vhf_last, "vj", None) is not None
    #     )
    #     if incremental_jk:
    #         _dm = np.asarray(dm) - np.asarray(dm_last)
    #     else:
    #         _dm = dm

    #     vj = self.get_j(mol, _dm, hermi)
    #     vk = None
    #     vxc = self.extra_vxc
    #     t0 = logger.timer(self, "jk", *t0)

    #     if ground_state:
    #         ecoul = np.einsum("ij,ji", dm, vj).real * 0.5
    #     else:
    #         ecoul = None

    #     if self.if_ZMP:
    #         # print("Using ZMP potential", flush=True)
    #         vxc += self.lambda_ZMP * (vj - self.J_tar)

    #     # if self.if_DMP:
    #     #     # print("Using DMP potential", flush=True)
    #     #     vxc += (
    #     #         2
    #     #         * self.lambda_DMP
    #     #         * self.s_half
    #     #         @ np.transpose(dm - self.dm_tar)
    #     #         @ self.s_half
    #     #     )

    #     vxc = lib.tag_array(vxc, ecoul=ecoul, exc=0, vj=vj, vk=vk)

    #     return vxc

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

        # self.kernel(self.dm)
        # self.dm = self.make_rdm1()
        # print(
        #     f"Use cycles = {self.cycles}, and final convergence = {self.converged}",
        #     flush=True,
        # )

        self.zdiis = DIIS(self.S, self.diis_space)

        for cycle in range(1, self.max_cycle):
            self.J = self.get_jk(self.mol, self.dm)[0]

            self.F = self.F0 + lambda_ZMP * (self.J - self.J_tar)
            self.F = pyscf.scf.hf.level_shift(
                self.S, self.dm * 0.5, self.F, self.level_shift
            )
            self.F, diis_e = self.zdiis.extrapolate(cycle, self.F, self.dm)  # DIIS

            self.mo_energy, self.mo_coeff = pyscf.scf.hf.eig(self.F, self.S)
            self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)

            ddm = self.dm_old - self.dm
            dm_e = np.max(np.abs(ddm))
            self.dm_old = self.dm
            dm_converged = dm_e < self.conv_tol_dm
            diis_converged = diis_e < self.conv_tol_diis
            self.mo_energy[self.mo_occ == 0] -= self.level_shift

            nocc = self.mol.nelectron // 2
            HOMO, LUMO = self.mo_energy[nocc - 1], self.mo_energy[nocc]
            gap = LUMO - HOMO

            print(
                f"\rlambda= {lambda_ZMP:7.2f}  iter: {cycle:4d} gap= {gap:10.7f}   ",
                end="\r",
            )

            self.converged = dm_converged and diis_converged
            if self.converged and cycle > 1:
                break


def uzmp(dm1_dft, dm1_cc):
    return
