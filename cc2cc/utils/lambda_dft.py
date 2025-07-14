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

    def __init__(self, mol, dm_tar, xc="LDA,VWN", lambda_rho=0.0):
        """
        Initialize the LambdaRKS class with a lambda parameter.

        :param lambda_: The scaling factor for the dipole potential contribution in the effective potential.
        """
        super().__init__(mol, xc=xc)
        self.grids.build()
        self.lambda_rho = lambda_rho
        self.dm_tar = dm_tar
        self.vxc = None

    def reset_vxc(self):
        """
        Reset the xc functional to a new one.
        """
        self.vxc = None

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """
        modified get_veff method to include a lambda parameter.
        """
        if self.vxc is None:
            self.vxc = get_veff_rks(
                self, mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi
            )
        delta_j = self.get_j(mol, dm - self.dm_tar, hermi=hermi)
        vxc_new = self.vxc + self.lambda_rho * delta_j

        vxc = lib.tag_array(
            vxc_new,
            ecoul=self.vxc.ecoul,
            exc=self.vxc.exc,
            vj=self.vxc.vj,
            vk=self.vxc.vk,
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
        assert dm_tar.ndim == 3, "dm_tar must be a 3D array"
        self.dm_tar = dm_tar
        self.vxc = None

    def reset_vxc(self):
        """
        Reset the xc functional to a new one.
        """
        self.vxc = None

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """
        modified get_veff method to include a lambda parameter.
        """
        if self.vxc is None:
            self.vxc = get_veff_uks(
                self, mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi
            )
        delta_j = self.get_j(mol, dm - self.dm_tar, hermi=hermi)
        vxc_new = self.vxc + self.lambda_rho * delta_j

        vxc = lib.tag_array(
            vxc_new,
            ecoul=self.vxc.ecoul,
            exc=self.vxc.exc,
            vj=self.vxc.vj,
            vk=self.vxc.vk,
        )

        return vxc
