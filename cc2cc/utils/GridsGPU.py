"""GPU-backed DFT grids for cc2cc.

This module mirrors :mod:`cc2cc.utils.Grids` but imports the grid and
numerical-integration implementation from ``gpu4pyscf`` instead of ``pyscf``.
It is intentionally kept separate from the CPU implementation so callers can
opt in to GPU4PySCF explicitly.
"""

from __future__ import annotations

import cupy as cp
from pyscf import __config__
from gpu4pyscf.dft import gen_grid, numint, radi
from gpu4pyscf.dft.gen_grid import BLKSIZE, NBINS, ALIGNMENT_UNIT
from gpu4pyscf.lib.cupy_helper import asarray

from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_LEN, EDGE_SIZE

GridsGPU = gen_grid.Grids
OCCDROP = getattr(__config__, "dft_numint_occdrop", 1e-12)
SWITCH_SIZE = getattr(__config__, "dft_numint_switch_size", 800)


def iterate_grid_segments(mol, grids, nao, deriv, max_memory):
    """Iterate grid blocks using GPU4PySCF grid arrays.

    The yielded ``coords`` and ``weight`` objects keep the backend used by
    ``grids`` (normally CuPy for GPU4PySCF grids).
    """
    ngrids = grids.coords.shape[0]
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    blksize = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
    blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ip0 + blksize, ngrids)
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        yield weight, coords


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
    """Evaluate density with GPU4PySCF ``NumInt``."""
    if getattr(dms, "mo_coeff", None) is not None:
        return ni.eval_rho2(
            mol, ao, dms.mo_coeff, dms.mo_occ, non0tab, xctype, with_lapl
        )
    return ni.eval_rho(mol, ao, dms, non0tab, xctype, hermi, with_lapl)


def gen_cube_cp(coords):
    """Generate full cube coordinates around every grid point on GPU."""
    idx = cp.arange(EDGE_SIZE, dtype=coords.dtype) - CUBE_MIDDLE
    ii, jj, kk = cp.meshgrid(idx, idx, idx, indexing="ij")
    offsets = cp.stack((ii.ravel(), jj.ravel(), kk.ravel()), axis=1) * EDGE_LEN

    coor_flat = coords[:, cp.newaxis, :] + offsets[cp.newaxis, :, :]
    return coor_flat.reshape((-1, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3))


def gen_cube5_cp(coords):
    """Generate the five-point cube representation around every grid point on GPU."""
    points = cp.array(
        [(0, 0, 0), (0, 2, 2), (1, 1, 1), (2, 2, 0), (2, 0, 2)],
        dtype=coords.dtype,
    )
    offsets = (points - CUBE_MIDDLE) * EDGE_LEN
    return coords[:, cp.newaxis, :] + offsets[cp.newaxis, :, :]


def gen_cube_njit(coords, coor_cube):
    """CPU-compatible signature that writes cube coordinates in-place."""
    coor_cube[...] = gen_cube_cp(coords)


def gen_cube5_njit(coords, coor_cube):
    """CPU-compatible signature that writes five-point coordinates in-place."""
    coor_cube[...] = gen_cube5_cp(coords)


class Grid(GridsGPU):
    """GPU4PySCF-backed version of :class:`cc2cc.utils.Grids.Grid`."""

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

        self.opt = None

    def gen_cube(self, mol, dms, coords):
        """Generate cube coordinates with GPU AO/rho contractions."""
        coords = asarray(coords)

        if self.cube_type == "center":
            coor_cube = coords.copy().reshape((len(coords), 1, 3))
        elif self.cube_type == "cube":
            coor_cube = cp.zeros((len(coords), EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3))
            gen_cube_njit(coords, coor_cube)
        elif self.cube_type == "cube5":
            coor_cube = cp.zeros((len(coords), 5, 3))
            gen_cube5_njit(coords, coor_cube)
        else:
            raise ValueError("Unknown cube type.")

        return GridCube(coor_cube, self, mol)


class GridCube:
    """Cube grids whose coordinates are stored as CuPy arrays for GPU4PySCF."""

    def __init__(self, coords, grid: Grid, mol):
        self.number_of_cube = len(coords)
        self.input_level = grid.input_level
        self.cube_type = grid.cube_type
        self.cube_size = grid.cube_size
        self.coords = asarray(coords).reshape((-1, 3))
        self.mol = mol
        self.cutoff = grid.cutoff
        self.opt = grid.opt

    def gen_cube_rho_rks(
        self,
        ni: numint.NumInt,
        dms,
        ao_deriv=1,
    ):
        """Generate RKS cube densities and XC inputs on the GPU."""
        input_mat = cp.zeros((self.input_level, len(self.coords)))
        vxc_mat = cp.zeros((self.input_level, 4, len(self.coords)))

        ao_value = numint.eval_ao(
            self.mol, self.coords, deriv=ao_deriv, transpose=False, gdftopt=self.opt
        )
        rho = rho_evaluator(ni, self.mol, ao_value[:4], dms, xctype="GGA")
        rho0 = rho[0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho0, xctype="LDA")[:2]
        input_mat[0] = exc_lda[:, 0] * rho0
        vxc_mat[0, 0:1, :] = vxc_lda

        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho0, xctype="LDA")[:2]
        input_mat[1] = exc_vwn[:, 0] * rho0
        vxc_mat[1, 0:1, :] = vxc_vwn

        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, xctype="GGA")[:2]
        input_mat[2] = exc_b88[:, 0] * rho0
        vxc_mat[2, :, :] = vxc_b88

        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, xctype="GGA")[:2]
        input_mat[3] = exc_lyp[:, 0] * rho0
        vxc_mat[3, :, :] = vxc_lyp

        if self.input_level > 4:
            exc_pbec, vxc_pbec = ni.eval_xc_eff("PBE,", rho, xctype="GGA")[:2]
            input_mat[4] = exc_pbec[:, 0] * rho0
            vxc_mat[4, :, :] = vxc_pbec

        if self.input_level > 5:
            exc_pbex, vxc_pbex = ni.eval_xc_eff(",PBE", rho, xctype="GGA")[:2]
            input_mat[5] = exc_pbex[:, 0] * rho0
            vxc_mat[5, :, :] = vxc_pbex

        if self.input_level > 6:
            exc_tfk, vxc_tfk = ni.eval_xc_eff("GGA_K_TFVW", rho, xctype="GGA")[:2]
            input_mat[6] = exc_tfk[:, 0] * rho0
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
        """Generate UKS cube densities and XC inputs on the GPU."""
        input_mat = cp.zeros((self.input_level, len(self.coords)))
        vxc_mat = cp.zeros((self.input_level, 2, 4, len(self.coords)))

        dma, dmb = dms

        ao_value = numint.eval_ao(
            self.mol, self.coords, deriv=ao_deriv, transpose=False, gdftopt=self.opt
        )
        rho_a = rho_evaluator(ni, self.mol, ao_value[:4], dma, xctype="GGA")
        rho_b = rho_evaluator(ni, self.mol, ao_value[:4], dmb, xctype="GGA")
        rho = cp.empty([2, 4, len(self.coords)])
        rho[0] = rho_a
        rho[1] = rho_b
        rho_lda = cp.empty([2, 1, len(self.coords)])
        rho_lda[0] = rho_a[0]
        rho_lda[1] = rho_b[0]
        rho0 = rho_a[0] + rho_b[0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, xctype="LDA")[:2]
        input_mat[0] = exc_lda[:, 0] * rho0
        vxc_mat[0, :, 0:1, :] = vxc_lda

        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, xctype="LDA")[:2]
        input_mat[1] = exc_vwn[:, 0] * rho0
        vxc_mat[1, :, 0:1, :] = vxc_vwn

        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, xctype="GGA")[:2]
        input_mat[2] = exc_b88[:, 0] * rho0
        vxc_mat[2, :, :, :] = vxc_b88

        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, xctype="GGA")[:2]
        input_mat[3] = exc_lyp[:, 0] * rho0
        vxc_mat[3, :, :, :] = vxc_lyp

        if self.input_level > 4:
            exc_pbec, vxc_pbec = ni.eval_xc_eff("PBE,", rho, xctype="GGA")[:2]
            input_mat[4] = exc_pbec[:, 0] * rho0
            vxc_mat[4, :, :, :] = vxc_pbec

        if self.input_level > 5:
            exc_pbex, vxc_pbex = ni.eval_xc_eff(",PBE", rho, xctype="GGA")[:2]
            input_mat[5] = exc_pbex[:, 0] * rho0
            vxc_mat[5, :, :, :] = vxc_pbex

        if self.input_level > 6:
            exc_tfk, vxc_tfk = ni.eval_xc_eff("GGA_K_TFVW", rho, xctype="GGA")[:2]
            input_mat[6] = exc_tfk[:, 0] * rho0
            vxc_mat[6, :, :, :] = vxc_tfk

        input_mat = input_mat.reshape(
            (self.input_level, self.number_of_cube, self.cube_size)
        ).transpose(1, 0, 2)
        vxc_mat = vxc_mat.reshape(
            (self.input_level, 2, 4, self.number_of_cube, self.cube_size)
        )

        return input_mat, vxc_mat, ao_value


GridGPU = Grid

__all__ = [
    "GridGPU",
]
