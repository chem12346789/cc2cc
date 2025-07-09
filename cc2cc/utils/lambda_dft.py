# File: cc2cc/utils/lambda_dft.py
# -*- coding: utf-8 -*-
"""
This module defines a class `LambdaRKS` and `LambdaUKS` that extends the PySCF.
Will return a close dm to the target dm, but with a modified effective potential.
"""
import numpy as np

import pyscf
from pyscf import lib

from pyscf.dft.rks import RKS, get_veff as get_veff_rks
from pyscf.dft.uks import UKS, get_veff as get_veff_uks


class LambdaRKS(RKS):
    """
    A class to handle the modified get_veff method for DFT calculations with a lambda parameter.
    This class modifies the get_veff method of the RKS class in PySCF to include a lambda parameter
    that can be used to scale the dipole potential contribution in the effective potential calculation.
    """

    def __init__(self, mol, dm_tar, xc="LDA,VWN", lambda_rho=0.0, lambda_dip=0.0):
        """
        Initialize the LambdaRKS class with a lambda parameter.

        :param lambda_: The scaling factor for the dipole potential contribution in the effective potential.
        """
        super().__init__(mol, xc=xc)
        self.grids.build()
        self.lambda_rho = lambda_rho
        self.lambda_dip = lambda_dip
        self.ao_0 = pyscf.dft.numint.eval_ao(mol, self.grids.coords)
        self.rho_tar = pyscf.dft.numint.eval_rho(mol, self.ao_0, dm_tar, xctype="LDA")

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """
        modified get_veff method to include a lambda parameter.
        """
        vxc = get_veff_rks(
            self, mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi
        )

        rho_dm = pyscf.dft.numint.eval_rho(mol, self.ao_0, dm, xctype="LDA")
        rho_diff = (
            (rho_dm - self.rho_tar) + (rho_dm - self.rho_tar) + (rho_dm - self.rho_tar)
        )
        dip_diff = (
            (rho_dm - self.rho_tar) * self.grids.coords[:, 0] ** 2
            + (rho_dm - self.rho_tar) * self.grids.coords[:, 1] ** 2
            + (rho_dm - self.rho_tar) * self.grids.coords[:, 2] ** 2
        )
        v_p_rho = pyscf.dft.numint.eval_mat(
            mol, self.ao_0, self.grids.weights, rho_diff, rho_diff
        )
        v_p_dip = pyscf.dft.numint.eval_mat(
            mol, self.ao_0, self.grids.weights, dip_diff, dip_diff
        )
        vxc_new = vxc + self.lambda_rho * v_p_rho + self.lambda_dip * v_p_dip

        vxc = lib.tag_array(
            vxc_new,
            ecoul=vxc.ecoul,
            exc=vxc.exc,
            vj=vxc.vj,
            vk=vxc.vk,
        )

        return vxc


class LambdaUKS(UKS):
    """
    A class to handle the modified get_veff method for DFT calculations with a lambda parameter.
    This class modifies the get_veff method of the RKS class in PySCF to include a lambda parameter
    that can be used to scale the dipole potential contribution in the effective potential calculation.
    """

    def __init__(self, mol, dm_tar, xc="LDA,VWN", lambda_rho=0.0, lambda_dip=0.0):
        """
        Initialize the LambdaRKS class with a lambda parameter.

        :param lambda_: The scaling factor for the dipole potential contribution in the effective potential.
        """
        super().__init__(mol, xc=xc)
        self.grids.build()
        self.lambda_rho = lambda_rho
        self.lambda_dip = lambda_dip
        self.ao_0 = pyscf.dft.numint.eval_ao(mol, self.grids.coords)
        assert dm_tar.ndim == 3, "dm_tar must be a 3D array"
        self.rho_a_tar = pyscf.dft.numint.eval_rho(
            mol, self.ao_0, dm_tar[0], xctype="LDA"
        )
        self.rho_b_tar = pyscf.dft.numint.eval_rho(
            mol, self.ao_0, dm_tar[1], xctype="LDA"
        )

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """
        modified get_veff method to include a lambda parameter.
        """
        vxc = get_veff_uks(
            self, mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi
        )

        rho_a_dm = pyscf.dft.numint.eval_rho(mol, self.ao_0, dm[0], xctype="LDA")
        rho_b_dm = pyscf.dft.numint.eval_rho(mol, self.ao_0, dm[1], xctype="LDA")
        rho_a_diff = (
            (rho_a_dm - self.rho_a_tar)
            + (rho_a_dm - self.rho_a_tar)
            + (rho_a_dm - self.rho_a_tar)
        )
        rho_b_diff = (
            (rho_b_dm - self.rho_b_tar)
            + (rho_b_dm - self.rho_b_tar)
            + (rho_b_dm - self.rho_b_tar)
        )
        dip_a_diff = (
            (rho_a_dm - self.rho_a_tar) * self.grids.coords[:, 0] ** 2
            + (rho_a_dm - self.rho_a_tar) * self.grids.coords[:, 1] ** 2
            + (rho_a_dm - self.rho_a_tar) * self.grids.coords[:, 2] ** 2
        )
        dip_b_diff = (
            (rho_b_dm - self.rho_b_tar) * self.grids.coords[:, 0] ** 2
            + (rho_b_dm - self.rho_b_tar) * self.grids.coords[:, 1] ** 2
            + (rho_b_dm - self.rho_b_tar) * self.grids.coords[:, 2] ** 2
        )
        v_p_a_rho = pyscf.dft.numint.eval_mat(
            mol, self.ao_0, self.grids.weights, rho_a_diff, rho_a_diff
        )
        v_p_b_rho = pyscf.dft.numint.eval_mat(
            mol, self.ao_0, self.grids.weights, rho_b_diff, rho_b_diff
        )
        v_p_a_dip = pyscf.dft.numint.eval_mat(
            mol, self.ao_0, self.grids.weights, dip_a_diff, dip_a_diff
        )
        v_p_b_dip = pyscf.dft.numint.eval_mat(
            mol, self.ao_0, self.grids.weights, dip_b_diff, dip_b_diff
        )
        vxc_new = (
            vxc
            + self.lambda_rho * np.array([v_p_a_rho, v_p_b_rho])
            + self.lambda_dip * np.array([v_p_a_dip, v_p_b_dip])
        )

        vxc = lib.tag_array(
            vxc_new,
            ecoul=vxc.ecoul,
            exc=vxc.exc,
            vj=vxc.vj,
            vk=vxc.vk,
        )

        return vxc
