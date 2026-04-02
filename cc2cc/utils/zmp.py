"""
ZMP is a method to invert the density matrix from CCSD(T). See the original paper for details: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.50.2138 and https://arxiv.org/pdf/2603.22140v1.
It is based on the idea of minimizing the difference between the DFT and CC density matrices in a least-squares sense.
"""

from functools import reduce

import numpy as np

import pyscf
from pyscf.lib import logger
import pyscf.dft
from pyscf.dft.numint import _dot_ao_ao_dense


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

    def __init__(self, mol, dm_tar, grids, faxc=0, dftxc=0, xc="b3lyp"):
        super().__init__(mol)
        self.initialize_grids(mol, dm_tar)

        self.faxc = faxc * (1.0 - 1.0 / self.mol.nelectron)
        print(f"Using faxc = {self.faxc:.2f} in RZMP", flush=True)
        self.dftxc = dftxc
        self.xc = xc
        self.dm_tar = dm_tar
        self.dm = dm_tar
        self.dm_old = dm_tar
        self.ao = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=0)
        self.weights = grids.weights
        self.rho_tar = pyscf.dft.numint.eval_rho(
            mol, self.ao, self.dm_tar, xctype="LDA"
        )
        self.J_tar = self.get_j(self.mol, self.dm_tar)
        self.S = mol.intor_symmetric("int1e_ovlp")

        self.conv_tol_dm = 1e-8
        self.conv_tol_diis = 1e-4

        J_tar, K_tar = self.get_jk(self.mol, self.dm_tar)
        ni = self._numint
        vxc = ni.nr_rks(self.mol, self.grids, self.xc, self.dm_tar)[2]
        self.extra_vxc = self.dftxc * vxc
        self.F0 = self.get_hcore(mol) + self.extra_vxc

        if not ni.libxc.is_hybrid_xc(self.xc):
            self.F0 = self.F0 + J_tar
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)

            if omega == 0:
                K_tar *= hyb
            elif alpha == 0:  # LR=0, only SR exchange
                raise NotImplementedError("LR=0 is not implemented yet")
            elif hyb == 0:  # SR=0, only LR exchange
                raise NotImplementedError("SR=0 is not implemented yet")
            self.F0 = self.F0 + J_tar - K_tar * 0.5

    def zscf(self, l=0.0):
        self.zdiis = DIIS(self.S, self.diis_space)

        t0 = (logger.process_clock(), logger.perf_counter())

        for cycle in range(1, self.max_cycle):
            self.J = self.get_jk(self.mol, self.dm)[0]

            self.F = self.F0
            self.F = self.F + l * (self.J - self.J_tar) + self.faxc * self.J

            rho_zmp = pyscf.dft.numint.eval_rho(
                self.mol, self.ao, self.dm, xctype="LDA"
            )
            wv = self.weights * (rho_zmp - self.rho_tar)
            self.F = self.F + l * _dot_ao_ao_dense(self.ao, self.ao, wv)

            self.F = pyscf.scf.hf.level_shift(
                self.S, self.dm * 0.5, self.F, self.level_shift
            )
            self.F, diis_e = self.zdiis.extrapolate(cycle, self.F, self.dm)

            self.mo_energy, self.mo_coeff = pyscf.scf.hf.eig(self.F, self.S)
            self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)

            ddm = self.dm_old - self.dm
            dm_e = np.max(np.abs(ddm))
            self.dm_old = self.dm
            self.mo_energy[self.mo_occ == 0] -= self.level_shift

            nocc = self.mol.nelectron // 2
            HOMO, LUMO = self.mo_energy[nocc - 1], self.mo_energy[nocc]
            gap = LUMO - HOMO

            t0 = logger.timer(self, f"{cycle:4d} gap {gap:.2e} dm_e {dm_e:.2e}", *t0)

            dm_converged = dm_e < self.conv_tol_dm
            diis_converged = diis_e < self.conv_tol_diis
            self.converged = dm_converged and diis_converged
            if self.converged and cycle > 1:
                break

        print(
            f"In lambda = {l:7.2f}, Use cycles = {cycle + 1}, and final convergence = {self.converged}",
            flush=True,
        )


class UZMP(pyscf.dft.uks.UKS):

    def __init__(self, mol, dm_tar, grids, faxc=0, dftxc=0, xc="b3lyp"):
        super().__init__(mol)
        self.initialize_grids(mol, dm_tar)

        self.faxc = faxc * (1.0 - 1.0 / self.mol.nelectron)
        print(f"Using faxc = {self.faxc:.2f} in UZMP", flush=True)
        self.dftxc = dftxc
        self.xc = xc
        self.dm_tar = dm_tar
        self.dm = dm_tar
        self.dm_old = dm_tar
        self.ao = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=0)
        self.weights = grids.weights
        self.rho_tar = [
            pyscf.dft.numint.eval_rho(mol, self.ao, self.dm_tar[0], xctype="LDA"),
            pyscf.dft.numint.eval_rho(mol, self.ao, self.dm_tar[1], xctype="LDA"),
        ]
        self.J_tar = self.get_j(self.mol, self.dm_tar)
        self.S = mol.intor_symmetric("int1e_ovlp")

        self.conv_tol_dm = 1e-8
        self.conv_tol_diis = 1e-4

        J_tar, K_tar = self.get_jk(self.mol, self.dm_tar)
        ni = self._numint
        vxc = ni.nr_uks(self.mol, self.grids, self.xc, self.dm_tar)[2]
        self.extra_vxc = self.dftxc * vxc
        self.F0 = self.get_hcore(mol) + self.extra_vxc

        if not ni.libxc.is_hybrid_xc(self.xc):
            self.F0 = self.F0 + J_tar[0] + J_tar[1]
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)

            if omega == 0:
                K_tar *= hyb
            elif alpha == 0:  # LR=0, only SR exchange
                raise NotImplementedError("LR=0 is not implemented yet")
            elif hyb == 0:  # SR=0, only LR exchange
                raise NotImplementedError("SR=0 is not implemented yet")
            self.F0 = self.F0 + J_tar[0] + J_tar[1] - K_tar * 0.5

    def zscf(self, l=0.0):
        self.zdiis_a = DIIS(self.S, self.diis_space)
        self.zdiis_b = DIIS(self.S, self.diis_space)

        t0 = (logger.process_clock(), logger.perf_counter())

        for cycle in range(1, self.max_cycle):
            J = self.get_jk(self.mol, self.dm)[0]

            Fa, Fb = self.F0
            Fa = Fa + 2 * l * (J[0] - self.J_tar[0]) + self.faxc * (J[0] + J[1])
            Fb = Fb + 2 * l * (J[1] - self.J_tar[1]) + self.faxc * (J[0] + J[1])

            rho_zmp = [
                pyscf.dft.numint.eval_rho(self.mol, self.ao, self.dm[0], xctype="LDA"),
                pyscf.dft.numint.eval_rho(self.mol, self.ao, self.dm[1], xctype="LDA"),
            ]
            wva = self.weights * (rho_zmp[0] - self.rho_tar[0])
            wvb = self.weights * (rho_zmp[1] - self.rho_tar[1])
            Fa = Fa + l * _dot_ao_ao_dense(self.ao, self.ao, wva)
            Fb = Fb + l * _dot_ao_ao_dense(self.ao, self.ao, wvb)

            Fa = pyscf.scf.hf.level_shift(self.S, self.dm[0], Fa, self.level_shift)
            Fb = pyscf.scf.hf.level_shift(self.S, self.dm[1], Fb, self.level_shift)
            Fa, diis_e_a = self.zdiis_a.extrapolate(cycle, Fa, self.dm[0])
            Fb, diis_e_b = self.zdiis_b.extrapolate(cycle, Fb, self.dm[1])

            e_a, c_a = pyscf.scf.hf.eig(Fa, self.S)
            e_b, c_b = pyscf.scf.hf.eig(Fb, self.S)
            self.mo_energy = np.array((e_a, e_b))
            self.mo_coeff = np.array((c_a, c_b))

            self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)

            ddm = self.dm_old - self.dm
            dm_e = np.max(np.abs(ddm))
            self.dm_old = self.dm
            self.mo_energy[0][self.mo_occ[0] == 0] -= self.level_shift
            self.mo_energy[1][self.mo_occ[1] == 0] -= self.level_shift

            HOMO = np.maximum(
                self.mo_energy[0][self.nelec[0] - 1],
                self.mo_energy[1][self.nelec[1] - 1],
            )
            LUMO = np.minimum(
                self.mo_energy[0][self.nelec[0]], self.mo_energy[1][self.nelec[1]]
            )
            gap = LUMO - HOMO

            t0 = logger.timer(self, f"{cycle:4d} gap {gap:.2e} dm_e {dm_e:.2e}", *t0)

            dm_converged = dm_e < self.conv_tol_dm
            diis_converged = diis_e_a + diis_e_b < self.conv_tol_diis
            self.converged = dm_converged and diis_converged
            if self.converged and cycle > 1:
                break

        print(
            f"In lambda = {l:7.2f}, Use cycles = {cycle + 1}, and final convergence = {self.converged}",
            flush=True,
        )
