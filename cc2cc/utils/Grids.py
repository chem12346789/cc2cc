"""CPU-backed DFT grids for cc2cc.

This module is the PySCF counterpart of :mod:`cc2cc.utils.grids_gpu`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pyscf import __config__
from pyscf.dft import gen_grid, numint, radi
from pyscf.lib import logger
from pyscf.dft.gen_grid import BLKSIZE, NBINS, ALIGNMENT_UNIT

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

GridsCPU = gen_grid.Grids
OCCDROP = getattr(__config__, "dft_numint_occdrop", 1e-12)
SWITCH_SIZE = getattr(__config__, "dft_numint_switch_size", 800)


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


@njit(fastmath=True)
def gen_cube_njit(rho_in_2, rho_in_1, coords, coor_cube):
    """Generate full cube coordinates around every grid point."""
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
def gen_cube5_njit(rho_in_2, rho_in_1, coords, coor_cube):
    """Generate the five-point cube representation around every grid point."""
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
            c0 = numint._dot_ao_dm(mol, ao[0], cpos, screen_index, shls_slice, ao_loc)
            c1 = numint._dot_ao_dm(mol, ao[1], cpos, screen_index, shls_slice, ao_loc)
            c2 = numint._dot_ao_dm(mol, ao[2], cpos, screen_index, shls_slice, ao_loc)
            c3 = numint._dot_ao_dm(mol, ao[3], cpos, screen_index, shls_slice, ao_loc)
            rho_in_1[0, :] += 2 * numint._contract_rho(c1, c0)
            rho_in_1[1, :] += 2 * numint._contract_rho(c2, c0)
            rho_in_1[2, :] += 2 * numint._contract_rho(c3, c0)

            c4 = numint._dot_ao_dm(mol, ao[4], cpos, screen_index, shls_slice, ao_loc)
            c5 = numint._dot_ao_dm(mol, ao[5], cpos, screen_index, shls_slice, ao_loc)
            c6 = numint._dot_ao_dm(mol, ao[6], cpos, screen_index, shls_slice, ao_loc)
            c7 = numint._dot_ao_dm(mol, ao[7], cpos, screen_index, shls_slice, ao_loc)
            c8 = numint._dot_ao_dm(mol, ao[8], cpos, screen_index, shls_slice, ao_loc)
            c9 = numint._dot_ao_dm(mol, ao[9], cpos, screen_index, shls_slice, ao_loc)

            rho_in_2[0, 0, :] += numint._contract_rho(c4, c0) + numint._contract_rho(
                c1, c1
            )
            rho_in_2[0, 1, :] += numint._contract_rho(c5, c0) + numint._contract_rho(
                c1, c2
            )
            rho_in_2[0, 2, :] += numint._contract_rho(c6, c0) + numint._contract_rho(
                c1, c3
            )
            rho_in_2[1, 1, :] += numint._contract_rho(c7, c0) + numint._contract_rho(
                c2, c2
            )
            rho_in_2[1, 2, :] += numint._contract_rho(c8, c0) + numint._contract_rho(
                c2, c3
            )
            rho_in_2[2, 2, :] += numint._contract_rho(c9, c0) + numint._contract_rho(
                c3, c3
            )
    else:
        c0 = numint._dot_ao_dm(mol, ao[0], dm, screen_index, shls_slice, ao_loc)
        rho_in_1[0, :] += 2 * numint._contract_rho(ao[1], c0)
        rho_in_1[1, :] += 2 * numint._contract_rho(ao[2], c0)
        rho_in_1[2, :] += 2 * numint._contract_rho(ao[3], c0)

        c1 = numint._dot_ao_dm(mol, ao[1], dm, screen_index, shls_slice, ao_loc)
        c2 = numint._dot_ao_dm(mol, ao[2], dm, screen_index, shls_slice, ao_loc)
        c3 = numint._dot_ao_dm(mol, ao[3], dm, screen_index, shls_slice, ao_loc)

        rho_in_2[0, 0, :] += numint._contract_rho(ao[4], c0) + numint._contract_rho(
            ao[1], c1
        )
        rho_in_2[0, 1, :] += numint._contract_rho(ao[5], c0) + numint._contract_rho(
            ao[1], c2
        )
        rho_in_2[0, 2, :] += numint._contract_rho(ao[6], c0) + numint._contract_rho(
            ao[1], c3
        )
        rho_in_2[1, 1, :] += numint._contract_rho(ao[7], c0) + numint._contract_rho(
            ao[2], c2
        )
        rho_in_2[1, 2, :] += numint._contract_rho(ao[8], c0) + numint._contract_rho(
            ao[2], c3
        )
        rho_in_2[2, 2, :] += numint._contract_rho(ao[9], c0) + numint._contract_rho(
            ao[3], c3
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
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()

        rho_in_1 = np.zeros((3, len(coords)))
        rho_in_2 = np.zeros((3, 3, len(coords)))
        ao = numint.eval_ao(mol, coords, deriv=2)

        if mol.spin == 0:
            eval_rho_cube(
                mol, ao, dms, rho_in_1, rho_in_2, screen_index, shls_slice, ao_loc
            )
        else:
            dma, dmb = numint._format_uks_dm(dms)
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
        ).transpose(1, 0, 2)
        vxc_mat = vxc_mat.reshape(
            (self.input_level, 4, self.number_of_cube, self.cube_size)
        )

        t0 = logger.timer(self.mol, "       gen exc and vxc", *t0)
        return input_mat, vxc_mat, ao_value

    def gen_cube_rho_uks(
        self,
        ni: numint.NumInt,
        dms,
        ao_deriv=1,
    ):
        """Generate UKS cube densities and XC inputs on the CPU."""
        t0 = (logger.process_clock(), logger.perf_counter())
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
        ).transpose(1, 0, 2)
        vxc_mat = vxc_mat.reshape(
            (self.input_level, 2, 4, self.number_of_cube, self.cube_size)
        )

        t0 = logger.timer(self.mol, "      gen exc and vxc", *t0)
        return input_mat, vxc_mat, ao_value


GridCPU = Grid
GridCubeCPU = GridCube

__all__ = [
    "GridCPU",
    "GridCubeCPU",
]
