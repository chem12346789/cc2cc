"""@package docstring
Documentation for this module.

More details.
"""

# pylint: disable=W0212

import numpy as np
import torch
from numba import njit, prange

import pyscf
from pyscf import dft, gto, lib
import pyscf.dft.numint
from pyscf.dft.numint import (
    _dot_ao_dm,
    _contract_rho,
    _sparse_enough,
    _empty_aligned,
    _format_uks_dm,
    eval_ao,
    _dot_ao_dm_sparse,
    _contract_rho_sparse,
)
from pyscf.dft.gen_grid import BLKSIZE, NBINS, CUTOFF, ALIGNMENT_UNIT
from pyscf import __config__

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_LEN, CUBE_MIDDLE

libdft = lib.load_library("libdft")
OCCDROP = getattr(__config__, "dft_numint_occdrop", 1e-12)
# The system size above which to consider the sparsity of the density matrix.
# If the number of AOs in the system is less than this value, all tensors are
# treated as dense quantities and contracted by dgemm directly.
SWITCH_SIZE = getattr(__config__, "dft_numint_switch_size", 800)


@njit(fastmath=True, parallel=True)
def gen_cube_njit(
    rho_in_2,
    rho_in_1,
    coords,
    coor_cube,
):
    """
    Generate the cube coordinates for the given molecule.
    """
    for p in prange(len(coords)):
        norm_2d = rho_in_2[:, :, p]
        eig_val, eig_vec = np.linalg.eigh(norm_2d)
        eig_val_sort = np.argsort(eig_val)
        eig_vec = eig_vec[:, eig_val_sort]
        norm_1d = rho_in_1[:, p]
        for i in range(3):
            if np.sum(eig_vec[:, i] * norm_1d) < 0:
                eig_vec[:, i] *= -1

        p_coords = coords[p]
        for i in range(CUBE_SIZE):
            for j in range(CUBE_SIZE):
                for k in range(CUBE_SIZE):
                    coor_cube[p, i, j, k, :] = (
                        p_coords
                        + (i - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 0]
                        + (j - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 1]
                        + (k - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 2]
                    )


@njit(fastmath=True, parallel=True)
def gen_center_njit(
    rho_in_2,
    rho_in_1,
    coords,
    coor_cube,
):
    """
    Generate the center coordinates for the given molecule.
    """
    coor_cube = coords.copy()


class Grid(dft.gen_grid.Grids):
    """
    Documentation for a class.

    This class is modified from pyscf.dft.gen_grid.Grids. Some default parameters are changed.
    New attributes:
    input_level:
        The input level for the grid generation.
        For example, 4 means the input is 4 energy density used by b3lyp functional.
    --------------------------------------------------------------------------
    Methods:
    gen_cube: Generate the cube coordinates for the given molecule.
    get_center_density: Get the center density of the cube.
    gen_cube_rho_rks: Generate the cube density for the given molecule in RKS.
    gen_cube_rho_uks: Generate the cube density for the given molecule in UKS.
    gen_rho_rks: Generate the center density for the given molecule in RKS.
    gen_rho_uks: Generate the center density for the given molecule in UKS.
    """

    def __init__(self, mol, level, input_level=4, cube_type="cube", test=False):
        super().__init__(mol)

        self.level = level
        self.input_level = input_level
        self.cube_type = cube_type

        # Set default parameters, please refer to pyscf.dft.gen_grid.Grids for details.
        self.radi_method = dft.radi.gauss_chebyshev
        self.becke_scheme = dft.gen_grid.original_becke
        self.atomic_radii = None
        self.radii_adjust = None
        if not test:
            self.prune = None
        self.build(with_non0tab=True, sort_grids=False)
        self.non0tab = self.make_mask(mol, self.coords)
        self.screen_index = self.non0tab

    def gen_cube(
        self,
        mol,
        dms,
        coords,
        screen_index=None,
    ):
        """
        Generate the cube coordinates for the given molecule.

        Args:
            mol: An instance of :class:`Mole'.
            dms: Density matrix, 2D array with shape (nao, nao). The orientation of the cube is determined by the eigenvectors of the Hessian matrix(secondary derivation of the density).
        """
        if mol.spin == 0:
            assert (
                np.linalg.norm(dms.conj().T - dms) < 1e-10
            ), "Density matrix is not symmetric."
            dm = dms
        else:
            assert (
                np.linalg.norm(dms[0].conj().T - dms[0]) < 1e-10
            ), "Density matrix is not symmetric."
            assert (
                np.linalg.norm(dms[1].conj().T - dms[1]) < 1e-10
            ), "Density matrix is not symmetric."
            dm = dms[0] + dms[1]

        # Hessian matrix
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()

        rho_in_1 = np.zeros((3, len(coords)))
        rho_in_2 = np.zeros((3, 3, len(coords)))
        ao = eval_ao(mol, coords, deriv=2)

        c0 = _dot_ao_dm(mol, ao[0], dm, screen_index, shls_slice, ao_loc)
        rho_in_1[0, :] = 2 * _contract_rho(ao[1], c0)
        rho_in_1[1, :] = 2 * _contract_rho(ao[2], c0)
        rho_in_1[2, :] = 2 * _contract_rho(ao[3], c0)

        c1 = _dot_ao_dm(mol, ao[1], dm, screen_index, shls_slice, ao_loc)
        c2 = _dot_ao_dm(mol, ao[2], dm, screen_index, shls_slice, ao_loc)
        c3 = _dot_ao_dm(mol, ao[3], dm, screen_index, shls_slice, ao_loc)

        rho_in_2[0, 0, :] = _contract_rho(ao[4], c0) + _contract_rho(ao[1], c1)
        rho_in_2[0, 1, :] = _contract_rho(ao[5], c0) + _contract_rho(ao[1], c2)
        rho_in_2[0, 2, :] = _contract_rho(ao[6], c0) + _contract_rho(ao[1], c3)
        rho_in_2[1, 1, :] = _contract_rho(ao[7], c0) + _contract_rho(ao[2], c2)
        rho_in_2[1, 2, :] = _contract_rho(ao[8], c0) + _contract_rho(ao[2], c3)
        rho_in_2[2, 2, :] = _contract_rho(ao[9], c0) + _contract_rho(ao[3], c3)

        rho_in_2[1, 0, :] = rho_in_2[0, 1, :]
        rho_in_2[2, 0, :] = rho_in_2[0, 2, :]
        rho_in_2[2, 1, :] = rho_in_2[1, 2, :]

        if self.cube_type == "center":
            coor_cube = np.zeros((len(coords), 1, 1, 1, 3))
            gen_center_njit(rho_in_2, rho_in_1, coords, coor_cube)
        elif self.cube_type == "cube":
            coor_cube = np.zeros((len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
            gen_cube_njit(rho_in_2, rho_in_1, coords, coor_cube)
        else:
            raise ValueError("Unknown cube type.")

        return GridCube(coor_cube, self)

    def gen_rho_rks(
        self,
        rho_in,
        ni,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        rho_lda = rho_in[0]
        rho0 = rho_in[0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho_in, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho_in, xctype="GGA")[:2]

        input_ = np.array(
            [exc_lda * rho0, exc_vwn * rho0, exc_b88 * rho0, exc_lyp * rho0]
        )
        input_ = input_.transpose(1, 0)

        if require_vxc:
            exc_b3lyp = (
                0.08 * input_[:, 0]
                + 0.19 * input_[:, 1]
                + 0.72 * input_[:, 2]
                + 0.81 * input_[:, 3]
            )

            vxc_b3lyp = np.zeros((4, 4, len(rho_lda)))
            vxc_b3lyp[0, 0:1, :] = vxc_lda
            vxc_b3lyp[1, 0:1, :] = vxc_vwn
            vxc_b3lyp[2, :, :] = vxc_b88
            vxc_b3lyp[3, :, :] = vxc_lyp

            return input_, exc_b3lyp, vxc_b3lyp

        return input_

    def gen_rho_uks(
        self,
        rho_in,
        ni,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        rho_lda = (rho_in[0][0], rho_in[1][0])
        rho0 = rho_in[0][0] + rho_in[1][0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho_in, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho_in, xctype="GGA")[:2]

        input_ = np.array(
            [exc_lda * rho0, exc_vwn * rho0, exc_b88 * rho0, exc_lyp * rho0]
        )
        input_ = input_.transpose(1, 0)

        if require_vxc:
            exc_b3lyp = (
                0.08 * input_[:, 0]
                + 0.19 * input_[:, 1]
                + 0.72 * input_[:, 2]
                + 0.81 * input_[:, 3]
            )

            vxc_b3lyp = np.zeros((4, 2, 4, len(rho_lda)))
            vxc_b3lyp[0, :, 0:1, :] = vxc_lda
            vxc_b3lyp[1, :, 0:1, :] = vxc_vwn
            vxc_b3lyp[2, :, :, :] = vxc_b88
            vxc_b3lyp[3, :, :, :] = vxc_lyp

            return input_, exc_b3lyp, vxc_b3lyp

        return input_


class GridCube:
    """
    Generate the Grids for the cube.
    Note that the no center weights are 0.
    The cutoff is the cutoff for the cube.
    """

    def __init__(self, coords, grid: Grid):
        self.number_of_cube = len(coords)
        # size of coords: (number * CUBE_SIZE * CUBE_SIZE * CUBE_SIZE, 3)
        self.input_level = grid.input_level
        self.coords = coords.reshape((-1, 3))
        self.mol = grid.mol
        self.cutoff = grid.cutoff
        self.non0tab = grid.make_mask(self.mol, self.coords)

    def gen_cube_rho_rks(self, ni: pyscf.dft.numint.NumInt, dms, require_vxc=False):
        """
        Generate the cube density for the given molecule.
        """
        input_mat = np.zeros((self.input_level, len(self.coords)))
        vxc_mat = np.zeros((self.input_level, 4, len(self.coords)))

        ao_value = ni.eval_ao(self.mol, self.coords, deriv=1, non0tab=self.non0tab)
        rho = ni.eval_rho(self.mol, ao_value, dms, non0tab=self.non0tab, xctype="GGA")
        rho0 = rho[0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho0, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho0, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, xctype="GGA")[:2]

        input_mat[0] = exc_lda * rho0
        input_mat[1] = exc_vwn * rho0
        input_mat[2] = exc_b88 * rho0
        input_mat[3] = exc_lyp * rho0

        vxc_mat[0, 0:1, :] = vxc_lda
        vxc_mat[1, 0:1, :] = vxc_vwn
        vxc_mat[2, :, :] = vxc_b88
        vxc_mat[3, :, :] = vxc_lyp

        if self.input_level > 4:
            exc_pbec, vxc_pbec = ni.eval_xc_eff("PBE,", rho, xctype="GGA")[:2]
            input_mat[4] = exc_pbec * rho0
            vxc_mat[4, :, :] = vxc_pbec

        if self.input_level > 5:
            exc_pbex, vxc_pbex = ni.eval_xc_eff(",PBE", rho, xctype="GGA")[:2]
            input_mat[5] = exc_pbex * rho0
            vxc_mat[5, :, :] = vxc_pbex

        if self.input_level > 6:
            exc_tfk, vxc_tfk = ni.eval_xc_eff("GGA_K_TFVW", rho, xctype="GGA")[:2]
            input_mat[6] = exc_tfk * rho0
            vxc_mat[6, :, :] = vxc_tfk

        input_mat = input_mat.reshape(
            (self.input_level, self.number_of_cube, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        )
        input_mat = input_mat.transpose(1, 0, 2, 3, 4)

        vxc_mat = vxc_mat.reshape(
            (self.input_level, 4, self.number_of_cube, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        )

        if require_vxc:
            return input_mat, vxc_mat, ao_value
        else:
            return input_mat

    def gen_cube_rho_uks(self, ni: pyscf.dft.numint.NumInt, dms, require_vxc=False):
        """
        Generate the cube density for the given molecule.
        """
        input_mat = np.zeros((self.input_level, len(self.coords)))
        vxc_mat = np.zeros((self.input_level, 2, 4, len(self.coords)))

        dma, dmb = _format_uks_dm(dms)

        ao_value = ni.eval_ao(self.mol, self.coords, deriv=1, non0tab=self.non0tab)
        rho_a = ni.eval_rho(self.mol, ao_value, dma, non0tab=self.non0tab, xctype="GGA")
        rho_b = ni.eval_rho(self.mol, ao_value, dmb, non0tab=self.non0tab, xctype="GGA")
        rho = (rho_a, rho_b)
        rho_lda = (rho_a[0], rho_b[0])
        rho0 = rho_a[0] + rho_b[0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, xctype="GGA")[:2]
        input_mat[0] = exc_lda * rho0
        input_mat[1] = exc_vwn * rho0
        input_mat[2] = exc_b88 * rho0
        input_mat[3] = exc_lyp * rho0

        vxc_mat[0, :, 0:1, :] = vxc_lda
        vxc_mat[1, :, 0:1, :] = vxc_vwn
        vxc_mat[2, :, :, :] = vxc_b88
        vxc_mat[3, :, :, :] = vxc_lyp

        if self.input_level > 4:
            exc_pbec, vxc_pbec = ni.eval_xc_eff("PBE,", rho, xctype="GGA")[:2]
            input_mat[4] = exc_pbec * rho0
            vxc_mat[4, :, :, :] = vxc_pbec

        if self.input_level > 5:
            exc_pbex, vxc_pbex = ni.eval_xc_eff(",PBE", rho, xctype="GGA")[:2]
            input_mat[5] = exc_pbex * rho0
            vxc_mat[5, :, :, :] = vxc_pbex

        if self.input_level > 6:
            exc_tfk, vxc_tfk = ni.eval_xc_eff("GGA_K_TFVW", rho, xctype="GGA")[:2]
            input_mat[6] = exc_tfk * rho0
            vxc_mat[6, :, :, :] = vxc_tfk

        input_mat = input_mat.reshape(
            (self.input_level, self.number_of_cube, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        )
        input_mat = input_mat.transpose(1, 0, 2, 3, 4)

        vxc_mat = vxc_mat.reshape(
            (
                self.input_level,
                2,
                4,
                self.number_of_cube,
                CUBE_SIZE,
                CUBE_SIZE,
                CUBE_SIZE,
            )
        )

        if require_vxc:
            return input_mat, vxc_mat, ao_value

        return input_mat
