"""@package docstring
Documentation for this module.

More details.
"""

# pylint: disable=W0212

import numpy as np
from numba import njit

import pyscf
from pyscf import dft, gto, lib
from pyscf.dft.numint import (
    _dot_ao_dm,
    _contract_rho,
    _sparse_enough,
    _empty_aligned,
    _format_uks_dm,
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


@njit(fastmath=True)
def gen_cube_njit(
    rho_input_2,
    rho_input_1,
    coords,
    coor_cube,
):
    """
    Generate the cube coordinates for the given molecule.
    """
    for p in range(len(coords)):
        norm_2d = rho_input_2[:, :, p]
        eig_val, eig_vec = np.linalg.eigh(norm_2d)
        eig_val_sort = np.argsort(eig_val)
        eig_vec = eig_vec[:, eig_val_sort]
        norm_1d = rho_input_1[:, p]
        for i in range(3):
            if eig_vec[:, i] @ norm_1d < 0:
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


def modified_block_loop(
    ni,
    mol,
    grids,
    nao=None,
    deriv=0,
    max_memory=25000,
    non0tab=None,
    blksize=None,
    buf=None,
):
    """Define this macro to loop over grids by blocks."""
    if grids.coords is None:
        grids.build(with_non0tab=True)
    if nao is None:
        nao = mol.nao
    ngrids = grids.coords.shape[0]
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    # NOTE to index grids.non0tab, the blksize needs to be an integer
    # multiplier of BLKSIZE
    if blksize is None:
        blksize = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
        blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    if non0tab is None and mol is grids.mol:
        non0tab = grids.non0tab
    if non0tab is None:
        non0tab = np.empty(
            ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
        )
        non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
    screen_index = non0tab

    # the xxx_sparse() functions require ngrids 8-byte aligned
    allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > SWITCH_SIZE

    if buf is None:
        buf = _empty_aligned(comp * blksize * nao)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[ip0:ip1]
        mask = screen_index[ip0 // BLKSIZE :]
        # TODO: pass grids.cutoff to eval_ao
        ao = ni.eval_ao(
            mol, coords, deriv=deriv, non0tab=mask, cutoff=grids.cutoff, out=buf
        )
        if not allow_sparse and not _sparse_enough(mask):
            # Unset mask for dense AO tensor. It determines which eval_rho
            # to be called in make_rho
            mask = None
        yield ao, mask, ip0, ip1


class GridCube:
    """
    Generate the Grids for the cube.
    Note that the no center weights are 0.
    The cutoff is the cutoff for the cube.
    This class is used to hack the modified_block_loop as the duck typing.
    """

    def __init__(self, coords, non0tab=None, cutoff=None):
        self.weights = None
        self.coords = coords.reshape((-1, 3))
        self.non0tab = non0tab
        self.mol = None
        self.cutoff = cutoff


class Grid(dft.gen_grid.Grids):
    """
    Documentation for a class.

    This class is modified from pyscf.dft.gen_grid.Grids. Some default parameters are changed.
    """

    def __init__(self, mol, level):
        super().__init__(mol)

        self.level = level
        self.atomic_radii = None
        self.radii_adjust = None
        self.radi_method = dft.radi.gauss_chebyshev
        self.becke_scheme = dft.gen_grid.original_becke
        self.prune = None
        self.build(with_non0tab=True, sort_grids=False)
        self.non0tab = self.make_mask(mol, self.coords)
        self.screen_index = self.non0tab

    def gen_cube(
        self,
        mol,
        dms,
        coords=None,
        screen_index=None,
    ):
        """
        Generate the cube coordinates for the given molecule.

        Args:
            mol: An instance of :class:`Mole'.
            dms: Density matrix, 2D array with shape (nao, nao). The orientation of the cube is determined by the eigenvectors of the Hessian matrix(secondary derivation of the density).
        """
        if getattr(dms, "mo_coeff", None) is None:
            print("Warning: dms.mo_coeff is None.")

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

        rho_input_1 = np.zeros((3, len(coords)))
        rho_input_2 = np.zeros((3, 3, len(coords)))
        ao = pyscf.dft.numint.eval_ao(mol, coords, deriv=2)

        c0 = _dot_ao_dm(mol, ao[0], dm, screen_index, shls_slice, ao_loc)
        rho_input_1[0, :] = 2 * _contract_rho(ao[1], c0)
        rho_input_1[1, :] = 2 * _contract_rho(ao[2], c0)
        rho_input_1[2, :] = 2 * _contract_rho(ao[3], c0)

        c1 = _dot_ao_dm(mol, ao[1], dm, screen_index, shls_slice, ao_loc)
        c2 = _dot_ao_dm(mol, ao[2], dm, screen_index, shls_slice, ao_loc)
        c3 = _dot_ao_dm(mol, ao[3], dm, screen_index, shls_slice, ao_loc)

        rho_input_2[0, 0, :] = _contract_rho(ao[4], c0) + _contract_rho(ao[1], c1)
        rho_input_2[0, 1, :] = _contract_rho(ao[5], c0) + _contract_rho(ao[1], c2)
        rho_input_2[0, 2, :] = _contract_rho(ao[6], c0) + _contract_rho(ao[1], c3)
        rho_input_2[1, 1, :] = _contract_rho(ao[7], c0) + _contract_rho(ao[2], c2)
        rho_input_2[1, 2, :] = _contract_rho(ao[8], c0) + _contract_rho(ao[2], c3)
        rho_input_2[2, 2, :] = _contract_rho(ao[9], c0) + _contract_rho(ao[3], c3)

        rho_input_2[1, 0, :] = rho_input_2[0, 1, :]
        rho_input_2[2, 0, :] = rho_input_2[0, 2, :]
        rho_input_2[2, 1, :] = rho_input_2[1, 2, :]

        coor_cube = np.zeros((len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
        gen_cube_njit(rho_input_2, rho_input_1, coords, coor_cube)

        return GridCube(coor_cube, cutoff=self.cutoff)

    def get_center_density(self, den_cube):
        """
        Get the center density of the cube.
        """
        return den_cube[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

    def gen_cube_rho_rks(
        self,
        rho_input,
        ni,
        dms,
        coords=None,
        mask=None,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        if coords is None:
            coords = self.coords
        if mask is None:
            mask = self.non0tab

        gridcube = self.gen_cube(self.mol, dms, coords, mask)

        input_ = np.zeros((4, len(coords) * CUBE_SIZE * CUBE_SIZE * CUBE_SIZE))
        make_rho, nset, nao = ni._gen_rho_evaluator(
            self.mol, dms, hermi, False, gridcube
        )
        for ao, mask, ip0, ip1 in modified_block_loop(
            ni, self.mol, gridcube, nao, deriv=1, non0tab=gridcube.non0tab
        ):
            for i in range(nset):
                rho = make_rho(i, ao, mask, ni._xc_type("b3lyp"))
                exc_lda = ni.eval_xc_eff("LDA,", rho[0], deriv=0, xctype="LDA")[0]
                exc_vwn = ni.eval_xc_eff(",VWN3", rho[0], deriv=0, xctype="LDA")[0]
                exc_b88 = ni.eval_xc_eff("B88,", rho, deriv=0, xctype="GGA")[0]
                exc_lyp = ni.eval_xc_eff(",LYP", rho, deriv=0, xctype="GGA")[0]
                rho0 = rho[0]
                input_[0, ip0:ip1] = exc_lda * rho0
                input_[1, ip0:ip1] = exc_vwn * rho0
                input_[2, ip0:ip1] = exc_b88 * rho0
                input_[3, ip0:ip1] = exc_lyp * rho0
        input_ = input_.reshape((4, len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        input_ = input_.transpose(1, 0, 2, 3, 4)

        if require_vxc:
            e_lda, v_lda = ni.eval_xc_eff("LDA,", rho_input[0], deriv=1, xctype="LDA")[
                :2
            ]
            e_vwn, v_vwn = ni.eval_xc_eff(",VWN3", rho_input[0], deriv=1, xctype="LDA")[
                :2
            ]
            e_b88, v_b88 = ni.eval_xc_eff("B88,", rho_input, deriv=1, xctype="GGA")[:2]
            e_lyp, v_lyp = ni.eval_xc_eff(",LYP", rho_input, deriv=1, xctype="GGA")[:2]

            exc_b3lyp = 0.08 * e_lda + 0.19 * e_vwn + 0.72 * e_b88 + 0.81 * e_lyp

            vxc_b3lyp = np.zeros((4, 4, len(coords)))
            vxc_b3lyp[0, 0:1, :] = v_lda
            vxc_b3lyp[1, 0:1, :] = v_vwn
            vxc_b3lyp[2, :, :] = v_b88
            vxc_b3lyp[3, :, :] = v_lyp
            return input_, exc_b3lyp * rho_input[0], vxc_b3lyp
        return input_

    def gen_cube_rho_uks(
        self,
        rho_input,
        ni,
        dms,
        coords=None,
        mask=None,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        if coords is None:
            coords = self.coords
        if mask is None:
            mask = self.non0tab

        gridcube = self.gen_cube(self.mol, dms, coords, mask)

        dma, dmb = _format_uks_dm(dms)
        nao = dma.shape[-1]
        make_rhoa, nset = ni._gen_rho_evaluator(self.mol, dma, hermi, False, gridcube)[
            :2
        ]
        make_rhob = ni._gen_rho_evaluator(self.mol, dmb, hermi, False, gridcube)[0]

        input_ = np.zeros((4, len(gridcube.coords)))

        for ao, mask, ip0, ip1 in modified_block_loop(ni, self.mol, gridcube, nao, 1):
            for i in range(nset):
                rho_a = make_rhoa(i, ao, mask, ni._xc_type("b3lyp"))
                rho_b = make_rhob(i, ao, mask, ni._xc_type("b3lyp"))
                rho = (rho_a, rho_b)
                rho_lda = (rho_a[0], rho_b[0])
                rho0 = rho_a[0] + rho_b[0]

                exc_lda = ni.eval_xc_eff("LDA,", rho_lda, deriv=0, xctype="LDA")[0]
                exc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, deriv=0, xctype="LDA")[0]
                exc_b88 = ni.eval_xc_eff("B88,", rho, deriv=0, xctype="GGA")[0]
                exc_lyp = ni.eval_xc_eff(",LYP", rho, deriv=0, xctype="GGA")[0]

                input_[:, ip0:ip1] = np.array(
                    [
                        exc_lda * rho0,
                        exc_vwn * rho0,
                        exc_b88 * rho0,
                        exc_lyp * rho0,
                    ]
                )
        del make_rhoa, make_rhob, nset

        input_ = input_.reshape((4, len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        input_ = input_.transpose(1, 0, 2, 3, 4)

        if require_vxc:
            rho_input_lda = (rho_input[0][0], rho_input[1][0])
            rho_input_0 = rho_input[0][0] + rho_input[1][0]

            e_lda, v_lda = ni.eval_xc_eff("LDA,", rho_input_lda, deriv=1, xctype="LDA")[
                :2
            ]
            e_vwn, v_vwn = ni.eval_xc_eff(
                ",VWN3", rho_input_lda, deriv=1, xctype="LDA"
            )[:2]
            e_b88, v_b88 = ni.eval_xc_eff("B88,", rho_input, deriv=1, xctype="GGA")[:2]
            e_lyp, v_lyp = ni.eval_xc_eff(",LYP", rho_input, deriv=1, xctype="GGA")[:2]

            exc_b3lyp = 0.08 * e_lda + 0.19 * e_vwn + 0.72 * e_b88 + 0.81 * e_lyp

            vxc_b3lyp = np.zeros((4, 2, 4, len(coords)))
            vxc_b3lyp[0, :, 0:1, :] = v_lda
            vxc_b3lyp[1, :, 0:1, :] = v_vwn
            vxc_b3lyp[2, :, :, :] = v_b88
            vxc_b3lyp[3, :, :, :] = v_lyp

            return input_, exc_b3lyp * rho_input_0, vxc_b3lyp

        return input_

    def gen_rho_rks(
        self,
        rho_input,
        ni,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        rho_lda = rho_input[0]
        rho0 = rho_input[0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho_input, deriv=1, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho_input, deriv=1, xctype="GGA")[:2]

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
        rho_input,
        ni,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        rho_lda = (rho_input[0][0], rho_input[1][0])
        rho0 = rho_input[0][0] + rho_input[1][0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho_input, deriv=1, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho_input, deriv=1, xctype="GGA")[:2]

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
