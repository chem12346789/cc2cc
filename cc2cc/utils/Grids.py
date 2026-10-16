"""CPU-backed DFT grids for cc2cc.

This module is the PySCF counterpart of :mod:`cc2cc.utils.grids_gpu`.
"""

from __future__ import annotations

import copy

import numpy as np
from numba import njit

import pyscf
from pyscf import __config__
from pyscf.dft import gen_grid, numint, radi
from pyscf.dft.gen_grid import BLKSIZE, NBINS, ALIGNMENT_UNIT
from pyscf.grad.rks import grids_response_cc

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

GridsCPU = gen_grid.Grids
OCCDROP = getattr(__config__, "dft_numint_occdrop", 1e-12)
SWITCH_SIZE = getattr(__config__, "dft_numint_switch_size", 800)
pyscf.lib.logger.TIMER_LEVEL = 4


def iterate_grid_segments(mol, grids, nao, deriv, max_memory, non0tab=None):
    ngrids = grids.coords.shape[0]
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    blksize = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
    blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    if mol is grids.mol:
        non0tab = grids.non0tab
    if non0tab is None:
        non0tab = np.empty(
            ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
        )
        non0tab[:] = NBINS + 1
    screen_index = non0tab

    allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > SWITCH_SIZE
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ip0 + blksize, ngrids)
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        mask = screen_index[ip0 // BLKSIZE :]
        if not allow_sparse:
            mask = None
        yield mask, weight, coords


def iterate_grid_response_segments(mol, grids, nao, deriv, max_memory):
    """Atom-centered blocks with Becke-weight nuclear derivatives."""
    for atom, (coords, weights, weight1) in enumerate(grids_response_cc(grids)):
        atom_grid = copy.copy(grids)
        atom_grid.coords = coords
        atom_grid.weights = weights
        atom_grid.non0tab = None
        start = 0
        for mask, block_weights, block_coords in iterate_grid_segments(
            mol, atom_grid, nao, deriv, max_memory
        ):
            stop = start + len(block_weights)
            yield atom, mask, block_weights, block_coords, weight1[:, :, start:stop]
            start = stop


def rho_evaluator(
    ni: numint.NumInt,
    mol,
    ao,
    dms,
    non0tab=None,
    xctype="LDA",
    hermi=0,
    with_lapl=True,
):
    """Evaluate density with PySCF ``NumInt``."""
    if getattr(dms, "mo_coeff", None) is not None:
        return ni.eval_rho2(
            mol, ao, dms.mo_coeff, dms.mo_occ, non0tab, xctype, with_lapl
        )
    return ni.eval_rho(mol, ao, dms, non0tab, xctype, hermi, with_lapl)


def gen_cube_njit(coords, coor_cube):
    """Generate full cube coordinates around every grid point."""
    coor_cube[:, :, :, :, :] = coords.reshape((len(coords), 1, 1, 1, 3))
    for i in range(EDGE_SIZE):
        for j in range(EDGE_SIZE):
            for k in range(EDGE_SIZE):
                coor_cube[:, i, j, k, :] += (
                    (i - CUBE_MIDDLE) * EDGE_LEN,
                    (j - CUBE_MIDDLE) * EDGE_LEN,
                    (k - CUBE_MIDDLE) * EDGE_LEN,
                )


@njit(fastmath=True)
def gen_cube5_njit(coords, coor_cube):
    """Generate the five-point cube representation around every grid point."""
    coor_cube[:, :, :, :, :] = coords.reshape((len(coords), 1, 1, 1, 3))
    for iter_, (i, j, k) in enumerate(
        [(0, 0, 0), (0, 2, 2), (1, 1, 1), (2, 2, 0), (2, 0, 2)]
    ):
        coor_cube[:, iter_, :, :, :] += (
            (i - CUBE_MIDDLE) * EDGE_LEN,
            (j - CUBE_MIDDLE) * EDGE_LEN,
            (k - CUBE_MIDDLE) * EDGE_LEN,
        )


class Grid(GridsCPU):
    """PySCF-backed version of :class:`cc2cc.utils.grids_gpu.Grid`."""

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
        self.radi_method = radi.gauss_chebyshev
        self.becke_scheme = gen_grid.original_becke
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
        """Generate cube coordinates with PySCF AO/rho contractions."""
        if self.cube_type == "center":
            coor_cube = coords.copy().reshape((len(coords), 1, 3))
        elif self.cube_type == "cube":
            coor_cube = np.zeros((len(coords), EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3))
            gen_cube_njit(coords, coor_cube)
        elif self.cube_type == "cube5":
            coor_cube = np.zeros((len(coords), 5, 3))
            gen_cube5_njit(coords, coor_cube)
        else:
            raise ValueError("Unknown cube type.")

        return GridCube(coor_cube, self)


class GridCube:
    """Cube grids whose coordinates are stored as NumPy arrays for PySCF."""

    def __init__(self, coords, grid: Grid):
        self.number_of_cube = len(coords)
        self.input_level = grid.input_level
        self.cube_type = grid.cube_type
        self.cube_size = grid.cube_size
        self.coords = coords.reshape((-1, 3))
        self.mol = grid.mol
        self.cutoff = grid.cutoff
        self.non0tab = None

    def gen_cube_rho_rks(
        self,
        ni: numint.NumInt,
        dms,
        ao_deriv=1,
    ):
        """Generate RKS cube densities and XC inputs on the CPU."""
        input_mat = np.zeros((self.input_level, len(self.coords)))
        vxc_mat = np.zeros((self.input_level, 4, len(self.coords)))

        ao_value = ni.eval_ao(
            self.mol, self.coords, deriv=ao_deriv, non0tab=self.non0tab
        )

        rho = rho_evaluator(
            ni, self.mol, ao_value[:4], dms, non0tab=self.non0tab, xctype="GGA"
        )
        rho0 = rho[0]

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
        ).transpose(1, 0, 2)
        vxc_mat = vxc_mat.reshape(
            (self.input_level, 4, self.number_of_cube, self.cube_size)
        )

        return input_mat, vxc_mat, ao_value

    def gen_cube_rho_uks(
        self,
        ni: numint.NumInt,
        dms,
        ao_deriv=1,
    ):
        """Generate UKS cube densities and XC inputs on the CPU."""
        input_mat = np.zeros((self.input_level, len(self.coords)))
        vxc_mat = np.zeros((self.input_level, 2, 4, len(self.coords)))

        dma, dmb = numint._format_uks_dm(dms)

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
        ).transpose(1, 0, 2)
        vxc_mat = vxc_mat.reshape(
            (self.input_level, 2, 4, self.number_of_cube, self.cube_size)
        )

        return input_mat, vxc_mat, ao_value


GridCPU = Grid

__all__ = [
    "GridCPU",
]
