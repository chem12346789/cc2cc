"""@package docstring
Documentation for this module.

More details.
"""

# pylint: disable=W0212

import ctypes
import gc
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
)
from pyscf.dft.gen_grid import BLKSIZE, NBINS, ALIGNMENT_UNIT
from pyscf import __config__

from cc2cc.utils.env_var import (
    CUBE_SIZE,
    CUBE_LEN,
    CUBE_MIDDLE,
    LEVEL,
    PERIOD,
)

libdft = lib.load_library("libdft")
OCCDROP = getattr(__config__, "dft_numint_occdrop", 1e-12)
# The system size above which to consider the sparsity of the density matrix.
# If the number of AOs in the system is less than this value, all tensors are
# treated as dense quantities and contracted by dgemm directly.
SWITCH_SIZE = getattr(__config__, "dft_numint_switch_size", 800)

libdft = lib.load_library("libdft")

LEBEDEV_ORDER = {
    0: 1,
    3: 6,
    5: 14,
    7: 26,
    9: 38,
    11: 50,
    13: 74,
    15: 86,
    17: 110,
    19: 146,
    21: 170,
    23: 194,
    25: 230,
    27: 266,
    29: 302,
    31: 350,
    35: 434,
    41: 590,
    47: 770,
    53: 974,
    59: 1202,
    65: 1454,
    71: 1730,
    77: 2030,
    83: 2354,
    89: 2702,
    95: 3074,
    101: 3470,
    107: 3890,
    113: 4334,
    119: 4802,
    125: 5294,
    131: 5810,
}

# fmt: off
# Period        1     2     3     4     5     6     7    # level
ANG_ORDER = np.array(
    (
        ( 0,   11,   15,   17,   17,   17,   17,   17),  # 0
        ( 0,   17,   23,   23,   23,   23,   23,   23),  # 1
        ( 0,   23,   29,   29,   29,   29,   29,   29),  # 2
        ( 0,   29,   29,   35,   35,   35,   35,   35),  # 3
        ( 0,   35,   41,   41,   41,   41,   41,   41),  # 4
        ( 0,   41,   47,   47,   47,   47,   47,   47),  # 5
        ( 0,   47,   53,   53,   53,   53,   53,   53),  # 6
        ( 0,   53,   59,   59,   59,   59,   59,   59),  # 7
        ( 0,   59,   59,   59,   59,   59,   59,   59),  # 8
        ( 0,   65,   65,   65,   65,   65,   65,   65),  # 9
    )
)

# Period        1     2     3     4     5     6     7   # level
RAD_GRIDS = np.array(
    (
        ( 0,   10,   15,   20,   30,   35,   40,   50), # 0
        ( 0,   30,   40,   50,   60,   65,   70,   75), # 1
        ( 0,   40,   60,   65,   75,   80,   85,   90), # 2
        ( 0,   50,   75,   80,   90,   95,  100,  105), # 3
        ( 0,   60,   90,   95,  105,  110,  115,  120), # 4
        ( 0,   70,  105,  110,  120,  125,  130,  135), # 5
        ( 0,   80,  120,  125,  135,  140,  145,  150), # 6
        ( 0,   90,  135,  140,  150,  155,  160,  165), # 7
        ( 0,  100,  150,  155,  165,  170,  175,  180), # 8
        ( 0,  200,  200,  200,  200,  200,  200,  200), # 9
    )
)
# fmt: on


def gen_atomic_grids(
    mol, atom_grid, radi_method=pyscf.dft.radi.gauss_chebyshev, **kwargs
):
    """
    Generate number of radial grids and angular grids for the given molecule.

    Returns:
        A dict, with the atom symbol for the dict key.  For each atom type,
        the dict value has two items: one is the meshgrid coordinates wrt the
        atom center; the second is the volume of that grid.
    """
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid) for ia in range(mol.natm)])
    atom_grids_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
            else:
                raise ValueError(
                    f"Atomic grid for {symb} is not defined. "
                    "Please provide a valid atom_grid."
                )
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

            rad_weight = 4 * np.pi * rad**2 * dr

            angs = [n_ang] * n_rad
            angs = np.array(angs)
            coords = []
            vol = []
            for n in sorted(set(angs)):
                grid = np.empty((n, 4))
                libdft.MakeAngularGrid(
                    grid.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n)
                )
                idx = np.where(angs == n)[0]
                coords.append(
                    np.einsum("i,jk->jik", rad[idx], grid[:, :3]).reshape(-1, 3)
                )
                vol.append(np.einsum("i,j->ji", rad_weight[idx], grid[:, 3]).ravel())
            atom_grids_tab[symb] = (np.vstack(coords), np.hstack(vol))
    return atom_grids_tab


def modified_build(grids, mol=None, **kwargs):
    """
    Build the grids with the given atomic grids.
    """
    if mol is None:
        mol = grids.mol
    atom_grids_tab = gen_atomic_grids(mol, grids.atom_grid, grids.radi_method, **kwargs)
    grids.coords, grids.weights = grids.get_partition(
        mol, atom_grids_tab, grids.radii_adjust, grids.atomic_radii, grids.becke_scheme
    )


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


class GridCube:
    """
    Generate the Grids for the cube.
    Note that the no center weights are 0.
    """

    def __init__(self, coords, weights):
        self.weights = np.zeros((coords.shape[0], CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        self.weights[:, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] = weights
        self.coords = coords.reshape(
            (weights.shape[0] * CUBE_SIZE * CUBE_SIZE * CUBE_SIZE, 3)
        )
        self.non0tab = None
        self.mol = None


class Grid(dft.gen_grid.Grids):
    """
    Documentation for a class.

    This class is modified from pyscf.dft.gen_grid.Grids. Some default parameters are changed.
    """

    def __init__(
        self,
        mol,
        level=LEVEL,
        period=PERIOD,
        n_rad=None,
        n_ang=None,
    ):
        super().__init__(mol)

        if n_rad is not None and n_ang is not None:
            self.n_rad = n_rad
            self.n_ang = n_ang
        else:
            self.n_rad, self.n_ang = (
                RAD_GRIDS[level, period],
                LEBEDEV_ORDER[ANG_ORDER[level, period]],
            )

        print(f"n_rad: {self.n_rad}, n_ang: {self.n_ang}")
        self.natm = mol.natm
        self.coord_list = []
        self.atom_grid = {}
        for i_atom in mol._atom:
            self.coord_list.append(i_atom[1:])
            self.atom_grid[i_atom[0]] = (self.n_rad, self.n_ang)
        self.coord_list = np.array(self.coord_list)

        self.prune = None
        self.atomic_radii = None
        self.radii_adjust = None
        self.becke_scheme = dft.gen_grid.original_becke
        self.radi_method = dft.radi.gauss_chebyshev
        modified_build(self)

    def gen_cube(
        self,
        mol,
        dm1_input,
        coords=None,
    ):
        """
        Generate the cube coordinates for the given molecule.

        Args:
            mol: An instance of :class:`Mole'.
            dm1_input: Density matrix, 2D array with shape (nao, nao). The orientation of the cube is determined by the eigenvectors of the Hessian matrix(secondary derivation of the density).
        """
        if coords is None:
            coords = self.coords

        ao_value = pyscf.dft.numint.eval_ao(mol, coords, deriv=2)

        # Hessian matrix
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        if mol.spin == 0:
            assert (
                np.linalg.norm(dm1_input.conj().T - dm1_input) < 1e-10
            ), "Density matrix is not symmetric."
            c0 = _dot_ao_dm(mol, ao_value[0], dm1_input, None, shls_slice, ao_loc)
        else:
            assert (
                np.linalg.norm(dm1_input[0].conj().T - dm1_input[0]) < 1e-10
            ), "Density matrix is not symmetric."
            assert (
                np.linalg.norm(dm1_input[1].conj().T - dm1_input[1]) < 1e-10
            ), "Density matrix is not symmetric."
            c0 = _dot_ao_dm(
                mol, ao_value[0], dm1_input[0] + dm1_input[1], None, shls_slice, ao_loc
            )

        rho_input_1 = np.zeros((3, len(coords)))
        rho_input_2 = np.zeros((3, 3, len(coords)))

        rho_input_1[0, :] = _contract_rho(ao_value[1], c0)
        rho_input_1[1, :] = _contract_rho(ao_value[2], c0)
        rho_input_1[2, :] = _contract_rho(ao_value[3], c0)
        rho_input_2[0, 0, :] = _contract_rho(ao_value[4], c0)
        rho_input_2[0, 1, :] = _contract_rho(ao_value[5], c0)
        rho_input_2[0, 2, :] = _contract_rho(ao_value[6], c0)
        rho_input_2[1, 1, :] = _contract_rho(ao_value[7], c0)
        rho_input_2[1, 2, :] = _contract_rho(ao_value[8], c0)
        rho_input_2[2, 2, :] = _contract_rho(ao_value[9], c0)
        rho_input_2[1, 0, :] = rho_input_2[0, 1, :]
        rho_input_2[2, 0, :] = rho_input_2[0, 2, :]
        rho_input_2[2, 1, :] = rho_input_2[1, 2, :]

        coor_cube = np.zeros((len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
        gen_cube_njit(rho_input_2, rho_input_1, coords, coor_cube)
        coor_cube = coor_cube.reshape(
            (len(coords) * CUBE_SIZE * CUBE_SIZE * CUBE_SIZE, 3)
        )
        return coor_cube

    def get_center_density(self, den_cube):
        """
        Get the center density of the cube.
        """
        return den_cube[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

    def gen_cube_rho_rks(
        self,
        mol,
        dms,
        rho,
        ni=None,
        coords=None,
        weights=None,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        if coords is None:
            coords = self.coords

        if weights is None:
            weights = self.weights

        coor_cube = self.gen_cube(mol, dms, coords)

        ao_cube = pyscf.dft.numint.eval_ao(mol, coor_cube, deriv=1)
        rho_cube = ni.eval_rho(mol, ao_cube, dms, xctype="GGA", hermi=hermi)
        exc_lda = ni.eval_xc_eff("LDA,", rho_cube[0], deriv=0, xctype="LDA")[0]
        exc_vwn = ni.eval_xc_eff(",VWN3", rho_cube[0], deriv=0, xctype="LDA")[0]
        exc_b88 = ni.eval_xc_eff("B88,", rho_cube, deriv=0, xctype="GGA")[0]
        exc_lyp = ni.eval_xc_eff(",LYP", rho_cube, deriv=0, xctype="GGA")[0]

        rho_cube0 = rho_cube[0]
        del ao_cube, rho_cube
        gc.collect()

        rho_input = np.array(
            [
                exc_lda * rho_cube0,
                exc_vwn * rho_cube0,
                exc_b88 * rho_cube0,
                exc_lyp * rho_cube0,
            ]
        )
        rho_input = rho_input.reshape((4, len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        rho_input = rho_input.transpose(1, 0, 2, 3, 4)
        del rho_cube0, exc_lda, exc_vwn, exc_b88, exc_lyp
        gc.collect()

        if require_vxc:
            rho_lda = rho[0]
            rho_0 = rho[0]
            e_lda, v_lda = ni.eval_xc_eff("LDA,", rho_lda, deriv=1, xctype="LDA")[:2]
            e_vwn, v_vwn = ni.eval_xc_eff(",VWN3", rho_lda, deriv=1, xctype="LDA")[:2]
            e_b88, v_b88 = ni.eval_xc_eff("B88,", rho, deriv=1, xctype="GGA")[:2]
            e_lyp, v_lyp = ni.eval_xc_eff(",LYP", rho, deriv=1, xctype="GGA")[:2]

            exc_b3lyp = 0.08 * e_lda + 0.19 * e_vwn + 0.72 * e_b88 + 0.81 * e_lyp
            vxc_b3lyp = np.zeros((4, 4, len(coords)))
            vxc_b3lyp[0, 0:1, :] = v_lda
            vxc_b3lyp[1, 0:1, :] = v_vwn
            vxc_b3lyp[2, :, :] = v_b88
            vxc_b3lyp[3, :, :] = v_lyp

            return exc_b3lyp * rho_0, rho_input, vxc_b3lyp

        return rho_input

    def gen_cube_rho_uks(
        self,
        mol,
        dms,
        rho,
        ni=None,
        coords=None,
        weights=None,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        if coords is None:
            coords = self.coords

        if weights is None:
            weights = self.weights

        coor_cube = self.gen_cube(mol, dms, coords)

        dma, dmb = _format_uks_dm(dms)

        ao_cube = pyscf.dft.numint.eval_ao(mol, coor_cube, deriv=1)
        rho_cube_a = pyscf.dft.numint.eval_rho(
            mol, ao_cube, dma, xctype="GGA", hermi=hermi
        )
        rho_cube_b = pyscf.dft.numint.eval_rho(
            mol, ao_cube, dmb, xctype="GGA", hermi=hermi
        )
        rho_cube = (rho_cube_a, rho_cube_b)
        rho_cube_lda = (rho_cube_a[0], rho_cube_b[0])

        exc_lda = ni.eval_xc_eff("LDA,", rho_cube_lda, deriv=0, xctype="LDA")[0]
        exc_vwn = ni.eval_xc_eff(",VWN3", rho_cube_lda, deriv=0, xctype="LDA")[0]
        exc_b88 = ni.eval_xc_eff("B88,", rho_cube, deriv=0, xctype="GGA")[0]
        exc_lyp = ni.eval_xc_eff(",LYP", rho_cube, deriv=0, xctype="GGA")[0]

        rho_cube0 = rho_cube_a[0] + rho_cube_b[0]
        del rho_cube_a, rho_cube_b, ao_cube
        gc.collect()

        rho_input = np.array(
            [
                exc_lda * rho_cube0,
                exc_vwn * rho_cube0,
                exc_b88 * rho_cube0,
                exc_lyp * rho_cube0,
            ]
        )
        del rho_cube0, exc_lda, exc_vwn, exc_b88, exc_lyp
        gc.collect()

        rho_input = rho_input.reshape((4, len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        rho_input = rho_input.transpose(1, 0, 2, 3, 4)

        if require_vxc:
            rho_lda = (rho[0][0], rho[1][0])
            rho_0 = rho[0][0] + rho[1][0]

            e_lda, v_lda = ni.eval_xc_eff("LDA,", rho_lda, deriv=1, xctype="LDA")[1]
            e_vwn, v_vwn = ni.eval_xc_eff(",VWN3", rho_lda, deriv=1, xctype="LDA")[1]
            e_b88, v_b88 = ni.eval_xc_eff("B88,", rho, deriv=1, xctype="GGA")[1]
            e_lyp, v_lyp = ni.eval_xc_eff(",LYP", rho, deriv=1, xctype="GGA")[1]

            exc_b3lyp = 0.08 * e_lda + 0.19 * e_vwn + 0.72 * e_b88 + 0.81 * e_lyp

            vxc_b3lyp = np.zeros((4, 2, 4, len(coords)))
            vxc_b3lyp[0, :, 0:1, :] = v_lda
            vxc_b3lyp[1, :, 0:1, :] = v_vwn
            vxc_b3lyp[2, :, :, :] = v_b88
            vxc_b3lyp[3, :, :, :] = v_lyp

            return exc_b3lyp * rho_0, rho_cube, vxc_b3lyp

        return rho_cube

    def gen_rho_rks(
        self,
        mol,
        dms,
        rho,
        ni=None,
        coords=None,
        weights=None,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        if coords is None:
            coords = self.coords

        if weights is None:
            weights = self.weights

        rho_lda = rho[0]
        rho_0 = rho[0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, deriv=1, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, deriv=1, xctype="GGA")[:2]

        rho_b3lyp = np.array(
            [exc_lda * rho_0, exc_vwn * rho_0, exc_b88 * rho_0, exc_lyp * rho_0]
        )

        if require_vxc:
            exc_b3lyp = (
                0.08 * rho_b3lyp[:, 0]
                + 0.19 * rho_b3lyp[:, 1]
                + 0.72 * rho_b3lyp[:, 2]
                + 0.81 * rho_b3lyp[:, 3]
            )

            vxc_b3lyp = np.zeros((4, 4, len(coords)))
            vxc_b3lyp[0, 0:1, :] = vxc_lda
            vxc_b3lyp[1, 0:1, :] = vxc_vwn
            vxc_b3lyp[2, :, :] = vxc_b88
            vxc_b3lyp[3, :, :] = vxc_lyp

            return exc_b3lyp, rho_b3lyp, vxc_b3lyp

        return rho_b3lyp

    def gen_rho_uks(
        self,
        mol,
        dms,
        rho,
        ni=None,
        coords=None,
        weights=None,
        hermi=1,
        require_vxc=False,
    ):
        """
        Generate the cube density for the given molecule.
        """
        if coords is None:
            coords = self.coords

        if weights is None:
            weights = self.weights

        rho_lda = (rho[0][0], rho[1][0])
        rho_0 = rho[0][0] + rho[1][0]

        exc_lda, vxc_lda = ni.eval_xc_eff("LDA,", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_vwn, vxc_vwn = ni.eval_xc_eff(",VWN3", rho_lda, deriv=1, xctype="LDA")[:2]
        exc_b88, vxc_b88 = ni.eval_xc_eff("B88,", rho, deriv=1, xctype="GGA")[:2]
        exc_lyp, vxc_lyp = ni.eval_xc_eff(",LYP", rho, deriv=1, xctype="GGA")[:2]

        rho_b3lyp = np.array(
            [exc_lda * rho_0, exc_vwn * rho_0, exc_b88 * rho_0, exc_lyp * rho_0]
        )

        if require_vxc:
            exc_b3lyp = (
                0.08 * rho_b3lyp[:, 0]
                + 0.19 * rho_b3lyp[:, 1]
                + 0.72 * rho_b3lyp[:, 2]
                + 0.81 * rho_b3lyp[:, 3]
            )

            vxc_b3lyp = np.zeros((4, 2, 4, len(coords)))
            vxc_b3lyp[0, :, 0:1, :] = vxc_lda
            vxc_b3lyp[1, :, 0:1, :] = vxc_vwn
            vxc_b3lyp[2, :, :, :] = vxc_b88
            vxc_b3lyp[3, :, :, :] = vxc_lyp

            return exc_b3lyp, rho_b3lyp, vxc_b3lyp

        return rho_b3lyp
