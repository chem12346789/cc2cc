"""@package docstring
Documentation for this module.

More details.
"""

import numpy as np
from numba import njit, prange

import pyscf
from pyscf import dft, lib
from pyscf.lib import logger
from pyscf.dft.gen_grid import Grids as GridsCPU
from pyscf.dft.gen_grid import BLKSIZE, NBINS, ALIGNMENT_UNIT
from pyscf.dft.numint import (
    NumInt,
    _dot_ao_dm,
    _contract_rho,
    _format_uks_dm,
    eval_ao,
    _sparse_enough,
)
from pyscf import __config__

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

libdft = lib.load_library("libdft")
OCCDROP = getattr(__config__, "dft_numint_occdrop", 1e-12)
# The system size above which to consider the sparsity of the density matrix.
# If the number of AOs in the system is less than this value, all tensors are
# treated as dense quantities and contracted by dgemm directly.
SWITCH_SIZE = getattr(__config__, "dft_numint_switch_size", 800)


def iterate_grid_segments(mol, grids, nao, deriv, max_memory, non0tab=None):
    ngrids = grids.coords.shape[0]
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    # NOTE to index grids.non0tab, the blksize needs to be an integer
    # multiplier of BLKSIZE
    blksize = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
    blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    if mol is grids.mol:
        non0tab = grids.non0tab
    if non0tab is None:
        non0tab = np.empty(
            ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
        )
        non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
    screen_index = non0tab

    # the xxx_sparse() functions require ngrids 8-byte aligned
    allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > SWITCH_SIZE

    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        mask = screen_index[ip0 // BLKSIZE :]
        if not allow_sparse and not _sparse_enough(mask):
            # Unset mask for dense AO tensor. It determines which eval_rho
            # to be called in make_rho
            mask = None
        yield mask, weight, coords


def rho_evaluator(
    ni: NumInt,
    mol,
    ao,
    dms,
    non0tab=None,
    xctype="LDA",
    hermi=0,
    with_lapl=True,
):
    if getattr(dms, "mo_coeff", None) is not None:
        # TODO: test whether dm.mo_coeff matching dm
        mo_coeff = dms.mo_coeff
        mo_occ = dms.mo_occ
    else:
        mo_coeff = mo_occ = None
    has_mo = mo_coeff is not None

    if has_mo:
        return ni.eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab, xctype, with_lapl)
    else:
        return ni.eval_rho(mol, ao, dms, non0tab, xctype, hermi, with_lapl)
        # it has a sparse version, but has_mo is False only for a few cases, so we use the dense version here.


@njit(fastmath=True)
def gen_cube_njit(
    rho_in_2,
    rho_in_1,
    coords,
    coor_cube,
):
    """
    Generate the cube coordinates for the given molecule.
    """
    for p in range(len(coords)):
        norm_2d = rho_in_2[:, :, p]
        eig_val, eig_vec = np.linalg.eigh(norm_2d)
        eig_val_sort = np.argsort(eig_val)
        eig_vec = eig_vec[:, eig_val_sort]
        norm_1d = rho_in_1[:, p]
        # norm_1d = np.array([np.pi, np.e, 1])
        for i in range(3):
            if (
                eig_vec[0, i] * norm_1d[0]
                + eig_vec[1, i] * norm_1d[1]
                + eig_vec[2, i] * norm_1d[2]
            ) < 0:
                eig_vec[:, i] *= -1

        p_coords = coords[p]
        for i in range(EDGE_SIZE):
            for j in range(EDGE_SIZE):
                for k in range(EDGE_SIZE):
                    coor_cube[p, i, j, k, :] = (
                        p_coords
                        + (i - CUBE_MIDDLE) * EDGE_LEN * eig_vec[:, 0]
                        + (j - CUBE_MIDDLE) * EDGE_LEN * eig_vec[:, 1]
                        + (k - CUBE_MIDDLE) * EDGE_LEN * eig_vec[:, 2]
                    )


@njit(fastmath=True)
def gen_cube5_njit(
    rho_in_2,
    rho_in_1,
    coords,
    coor_cube,
):
    """
    Generate the center coordinates for the given molecule.
    """
    for p in range(len(coords)):
        norm_2d = rho_in_2[:, :, p]
        eig_val, eig_vec = np.linalg.eigh(norm_2d)
        eig_val_sort = np.argsort(eig_val)
        eig_vec = eig_vec[:, eig_val_sort]
        norm_1d = rho_in_1[:, p]
        for i in range(3):
            if (
                eig_vec[0, i] * norm_1d[0]
                + eig_vec[1, i] * norm_1d[1]
                + eig_vec[2, i] * norm_1d[2]
            ) < 0:
                eig_vec[:, i] *= -1

        p_coords = coords[p]
        for iter_, (i, j, k) in enumerate(
            [(0, 0, 0), (0, 2, 2), (1, 1, 1), (2, 2, 0), (2, 0, 2)]
        ):
            coor_cube[p, iter_, :] = (
                p_coords
                + (i - CUBE_MIDDLE) * EDGE_LEN * eig_vec[:, 0]
                + (j - CUBE_MIDDLE) * EDGE_LEN * eig_vec[:, 1]
                + (k - CUBE_MIDDLE) * EDGE_LEN * eig_vec[:, 2]
            )


def eval_rho_cube(mol, ao, dm, rho_in_1, rho_in_2, screen_index, shls_slice, ao_loc):
    if getattr(dm, "mo_coeff", None) is not None:
        # TODO: test whether dm.mo_coeff matching dm
        mo_coeff = dm.mo_coeff
        mo_occ = dm.mo_occ
    else:
        mo_coeff = mo_occ = None
    has_mo = mo_coeff is not None

    if has_mo:
        pos = mo_occ > OCCDROP
        if np.any(pos):
            cpos = np.einsum("ij,j->ij", mo_coeff[:, pos], np.sqrt(mo_occ[pos]))
            c0 = _dot_ao_dm(mol, ao[0], cpos, screen_index, shls_slice, ao_loc)
            c1 = _dot_ao_dm(mol, ao[1], cpos, screen_index, shls_slice, ao_loc)
            c2 = _dot_ao_dm(mol, ao[2], cpos, screen_index, shls_slice, ao_loc)
            c3 = _dot_ao_dm(mol, ao[3], cpos, screen_index, shls_slice, ao_loc)
            rho_in_1[0, :] += 2 * _contract_rho(c1, c0)
            rho_in_1[1, :] += 2 * _contract_rho(c2, c0)
            rho_in_1[2, :] += 2 * _contract_rho(c3, c0)

            c4 = _dot_ao_dm(mol, ao[4], cpos, screen_index, shls_slice, ao_loc)
            c5 = _dot_ao_dm(mol, ao[5], cpos, screen_index, shls_slice, ao_loc)
            c6 = _dot_ao_dm(mol, ao[6], cpos, screen_index, shls_slice, ao_loc)
            c7 = _dot_ao_dm(mol, ao[7], cpos, screen_index, shls_slice, ao_loc)
            c8 = _dot_ao_dm(mol, ao[8], cpos, screen_index, shls_slice, ao_loc)
            c9 = _dot_ao_dm(mol, ao[9], cpos, screen_index, shls_slice, ao_loc)

            rho_in_2[0, 0, :] += _contract_rho(c4, c0) + _contract_rho(c1, c1)
            rho_in_2[0, 1, :] += _contract_rho(c5, c0) + _contract_rho(c1, c2)
            rho_in_2[0, 2, :] += _contract_rho(c6, c0) + _contract_rho(c1, c3)
            rho_in_2[1, 1, :] += _contract_rho(c7, c0) + _contract_rho(c2, c2)
            rho_in_2[1, 2, :] += _contract_rho(c8, c0) + _contract_rho(c2, c3)
            rho_in_2[2, 2, :] += _contract_rho(c9, c0) + _contract_rho(c3, c3)
    else:
        c0 = _dot_ao_dm(mol, ao[0], dm, screen_index, shls_slice, ao_loc)
        rho_in_1[0, :] += 2 * _contract_rho(ao[1], c0)
        rho_in_1[1, :] += 2 * _contract_rho(ao[2], c0)
        rho_in_1[2, :] += 2 * _contract_rho(ao[3], c0)

        c1 = _dot_ao_dm(mol, ao[1], dm, screen_index, shls_slice, ao_loc)
        c2 = _dot_ao_dm(mol, ao[2], dm, screen_index, shls_slice, ao_loc)
        c3 = _dot_ao_dm(mol, ao[3], dm, screen_index, shls_slice, ao_loc)

        rho_in_2[0, 0, :] += _contract_rho(ao[4], c0) + _contract_rho(ao[1], c1)
        rho_in_2[0, 1, :] += _contract_rho(ao[5], c0) + _contract_rho(ao[1], c2)
        rho_in_2[0, 2, :] += _contract_rho(ao[6], c0) + _contract_rho(ao[1], c3)
        rho_in_2[1, 1, :] += _contract_rho(ao[7], c0) + _contract_rho(ao[2], c2)
        rho_in_2[1, 2, :] += _contract_rho(ao[8], c0) + _contract_rho(ao[2], c3)
        rho_in_2[2, 2, :] += _contract_rho(ao[9], c0) + _contract_rho(ao[3], c3)


class Grid(GridsCPU):
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

    def __init__(
        self,
        mol,
        level,
        input_level=4,
        cube_type="cube",
        cube_size=EDGE_SIZE**3,
    ):
        super().__init__(mol)

        self.level = level
        self.input_level = input_level
        self.cube_type = cube_type
        self.cube_size = cube_size

        # Set default parameters, please refer to pyscf.dft.gen_grid.Grids for details.
        self.radi_method = dft.radi.gauss_chebyshev
        self.becke_scheme = dft.gen_grid.original_becke
        self.atomic_radii = None
        self.radii_adjust = None
        self.prune = None
        self.build(with_non0tab=False, sort_grids=False)
        self.non0tab = None
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

        # Hessian matrix
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()

        rho_in_1 = np.zeros((3, len(coords)))
        rho_in_2 = np.zeros((3, 3, len(coords)))
        ao = eval_ao(mol, coords, deriv=2)

        if mol.spin == 0:
            eval_rho_cube(
                mol, ao, dms, rho_in_1, rho_in_2, screen_index, shls_slice, ao_loc
            )
        else:
            dma, dmb = _format_uks_dm(dms)
            eval_rho_cube(
                mol, ao, dma, rho_in_1, rho_in_2, screen_index, shls_slice, ao_loc
            )
            eval_rho_cube(
                mol, ao, dmb, rho_in_1, rho_in_2, screen_index, shls_slice, ao_loc
            )

        rho_in_2[1, 0, :] = rho_in_2[0, 1, :]
        rho_in_2[2, 0, :] = rho_in_2[0, 2, :]
        rho_in_2[2, 1, :] = rho_in_2[1, 2, :]

        if self.cube_type == "center":
            coor_cube = coords.copy().reshape((len(coords), 1, 3))
        elif self.cube_type == "cube":
            coor_cube = np.zeros((len(coords), EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3))
            gen_cube_njit(rho_in_2, rho_in_1, coords, coor_cube)
        elif self.cube_type == "cube5":
            coor_cube = np.zeros((len(coords), 5, 3))
            gen_cube5_njit(rho_in_2, rho_in_1, coords, coor_cube)
        else:
            raise ValueError("Unknown cube type.")

        return GridCube(coor_cube, self)


class GridCube:
    """
    Generate the Grids for the cube.
    Note that the no center weights are 0.
    The cutoff is the cutoff for the cube.
    """

    def __init__(self, coords, grid: Grid):
        self.number_of_cube = len(coords)
        # size of coords: (number * EDGE_SIZE * EDGE_SIZE * EDGE_SIZE, 3) or (number * 5, 3)
        self.input_level = grid.input_level
        self.cube_type = grid.cube_type
        self.cube_size = grid.cube_size
        self.coords = coords.reshape((-1, 3))
        self.mol = grid.mol
        self.cutoff = grid.cutoff
        # sparse version seems to be not pallelized, and is slower than dense version.
        # So we use dense version here (set non0tab tobe None) for all system sizes.
        self.non0tab = None

    def gen_cube_rho_rks(
        self,
        ni: pyscf.dft.numint.NumInt,
        dms,
        ao_deriv=1,
    ):
        """
        Generate the cube density for the given molecule.
        """
        t0 = (logger.process_clock(), logger.perf_counter())
        t1 = (logger.process_clock(), logger.perf_counter())
        input_mat = np.zeros((self.input_level, len(self.coords)))
        vxc_mat = np.zeros((self.input_level, 4, len(self.coords)))

        ao_value = ni.eval_ao(
            self.mol, self.coords, deriv=ao_deriv, non0tab=self.non0tab
        )
        t1 = logger.timer(self.mol, "           ao_value", *t1)

        rho = rho_evaluator(
            ni, self.mol, ao_value[:4], dms, non0tab=self.non0tab, xctype="GGA"
        )
        rho0 = rho[0]
        t1 = logger.timer(self.mol, "           eval_rho", *t1)
        t0 = logger.timer(self.mol, "       gen input", *t0)

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho0, xctype="LDA")[:2]
        input_mat[0] = exc_lda * rho0
        vxc_mat[0, 0:1, :] = vxc_lda

        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho0, xctype="LDA")[:2]
        input_mat[1] = exc_vwn * rho0
        vxc_mat[1, 0:1, :] = vxc_vwn

        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, xctype="GGA")[:2]
        input_mat[2] = exc_b88 * rho0
        vxc_mat[2, :, :] = vxc_b88

        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, xctype="GGA")[:2]
        input_mat[3] = exc_lyp * rho0
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
            (self.input_level, self.number_of_cube, self.cube_size)
        )
        input_mat = input_mat.transpose(1, 0, 2)
        vxc_mat = vxc_mat.reshape(
            (self.input_level, 4, self.number_of_cube, self.cube_size)
        )

        t0 = logger.timer(self.mol, "       gen exc and vxc", *t0)
        return input_mat, vxc_mat, ao_value

    def gen_cube_rho_uks(
        self,
        ni: pyscf.dft.numint.NumInt,
        dms,
        ao_deriv=1,
    ):
        """
        Generate the cube density for the given molecule.
        """
        t0 = (logger.process_clock(), logger.perf_counter())
        input_mat = np.zeros((self.input_level, len(self.coords)))
        vxc_mat = np.zeros((self.input_level, 2, 4, len(self.coords)))

        dma, dmb = _format_uks_dm(dms)

        ao_value = ni.eval_ao(
            self.mol, self.coords, deriv=ao_deriv, non0tab=self.non0tab
        )
        rho_a = rho_evaluator(
            ni, self.mol, ao_value[:4], dma, non0tab=self.non0tab, xctype="GGA"
        )
        rho_b = rho_evaluator(
            ni, self.mol, ao_value[:4], dmb, non0tab=self.non0tab, xctype="GGA"
        )
        rho = (rho_a, rho_b)
        rho_lda = (rho_a[0], rho_b[0])
        rho0 = rho_a[0] + rho_b[0]
        t0 = logger.timer(self.mol, "      gen input", *t0)

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, xctype="LDA")[:2]
        input_mat[0] = exc_lda * rho0
        vxc_mat[0, :, 0:1, :] = vxc_lda

        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, xctype="LDA")[:2]
        input_mat[1] = exc_vwn * rho0
        vxc_mat[1, :, 0:1, :] = vxc_vwn

        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, xctype="GGA")[:2]
        input_mat[2] = exc_b88 * rho0
        vxc_mat[2, :, :, :] = vxc_b88

        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, xctype="GGA")[:2]
        input_mat[3] = exc_lyp * rho0
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
            (self.input_level, self.number_of_cube, self.cube_size)
        )
        input_mat = input_mat.transpose(1, 0, 2)
        vxc_mat = vxc_mat.reshape(
            (self.input_level, 2, 4, self.number_of_cube, self.cube_size)
        )

        t0 = logger.timer(self.mol, "      gen exc and vxc", *t0)
        return input_mat, vxc_mat, ao_value
