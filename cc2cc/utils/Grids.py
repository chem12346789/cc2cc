"""@package docstring
Documentation for this module.
 
More details.
"""

import ctypes
from itertools import product
import os

import numpy as np
from joblib import Parallel, parallel_config, delayed
import pyscf
from pyscf import dft, gto, lib
from pyscf.dft.numint import _dot_ao_dm, _contract_rho

from cc2cc.utils.env_var import (
    CUBE_SIZE,
    CUBE_LEN,
    CUBE_MIDDLE,
    LEVEL,
    PERIOD,
)

libdft = lib.load_library("libdft")

LEN = 7

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


def gen_input(rho, spin, xc_type):
    """
    Generate the input for the model.
    """
    if spin != 0:
        rho01, dx1, dy1, dz1 = rho[0][:4]
        rho02, dx2, dy2, dz2 = rho[1][:4]
        gamma1 = dx1**2 + dy1**2 + dz1**2
        gamma2 = dx2**2 + dy2**2 + dz2**2
        gamma12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2
    else:
        rho0, dx, dy, dz = rho[:4]
        gamma1 = gamma2 = gamma12 = (dx**2 + dy**2 + dz**2) / 4
        rho01 = rho02 = rho0 / 2

    if xc_type == "GGA":
        rho0 = np.array([rho01, rho02, gamma1, gamma12, gamma2])
    elif xc_type == "MGGA":
        if spin != 0:
            tau1 = rho[0][4]
            tau2 = rho[1][4]
        else:
            tau = rho[4]
            tau1 = tau * 0.5
            tau2 = tau * 0.5
        rho0 = np.array([rho01, rho02, gamma1, gamma12, gamma2, tau1, tau2])

    return rho0


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


class Grid(dft.gen_grid.Grids):
    """
    Documentation for a class.

    This class is modified from pyscf.dft.gen_grid.Grids. Some default parameters are changed.
    """

    def __init__(self, mol, level=LEVEL, period=PERIOD):
        super().__init__(mol)
        self.n_rad, self.n_ang = (
            RAD_GRIDS[level, period],
            LEBEDEV_ORDER[ANG_ORDER[level, period]],
        )
        print(f"n_rad: {self.n_rad}, n_ang: {self.n_ang}")
        self.natm = mol.natm
        self.coord_list = []
        self.atom_grid = {}
        for i_atom in mol.atom:
            self.coord_list.append(i_atom[1:])
            self.atom_grid[i_atom[0]] = (self.n_rad, self.n_ang)
        self.coord_list = np.array(self.coord_list)

        self.prune = None
        self.atomic_radii = None
        self.radii_adjust = None
        self.becke_scheme = dft.gen_grid.original_becke
        self.radi_method = dft.radi.gauss_chebyshev
        modified_build(self)

        self.coor_cube = None

    def gen_cube(self, mol, dm1_input):
        """
        Generate the cube coordinates for the given molecule.

        Args:
            mol: An instance of :class:`Mole'.
            dm1_input: Density matrix, 2D array with shape (nao, nao). The orientation of the cube is determined by the eigenvectors of the Hessian matrix(secondary derivation of the density).
        """
        if self.coor_cube is not None:
            print("Error: self.coor_cube is initialized!")
            return

        ao_value = pyscf.dft.numint.eval_ao(mol, self.coords, deriv=2)

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

        rho_input_1 = np.zeros((3, len(self.coords)))
        rho_input_2 = np.zeros((3, 3, len(self.coords)))

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

        def gen_cube_p(p):
            norm_2d = rho_input_2[:, :, p]
            eig_val, eig_vec = np.linalg.eigh(norm_2d)
            eig_val_sort = np.argsort(eig_val)
            eig_vec = eig_vec[:, eig_val_sort]
            norm_1d = rho_input_1[:, p]
            for i in range(3):
                if eig_vec[:, i] @ norm_1d < 0:
                    eig_vec[:, i] *= -1

            p_coords = self.coords[p]
            coords_cube = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
            for i, j, k in product(range(CUBE_SIZE), repeat=3):
                coords_cube[i, j, k, :] = (
                    p_coords
                    + (i - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 0]
                    + (j - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 1]
                    + (k - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 2]
                )
            return coords_cube

        with parallel_config(
            backend="loky",
            n_jobs=int(os.environ.get("OMP_NUM_THREADS", 6)),
            inner_max_num_threads=1,
        ):
            self.coor_cube = Parallel()(
                delayed(gen_cube_p)(p) for p in range(len(self.coords))
            )
        self.coor_cube = np.array(self.coor_cube)

    def gen_cube_mask(self):
        for p_coor_cube in self.coor_cube:
            for i_coor_cube in p_coor_cube:
                mask = np.where(np.linspace.norm(i_coor_cube - self.coords) < 1e-10)
                print(mask)

    def gen_cube_rho(self, mol, dm1_input, reset=False, xc_type="MGGA"):
        """
        Generate the cube density for the given molecule.
        """
        if self.coor_cube is None:
            print("Warning: coor_cube is not initialized!")
            self.gen_cube(mol, dm1_input)
        elif reset:
            self.gen_cube(mol, dm1_input)

        def gen_cube_rho_p(p):
            ao_cube = pyscf.dft.numint.eval_ao(
                mol, self.coor_cube[p].reshape(-1, 3), deriv=2
            )
            if mol.spin == 0:
                rho_cube_p = pyscf.dft.numint.eval_rho(
                    mol, ao_cube, dm1_input, xctype="mGGA", with_lapl=False
                )
                rho_cube_p_norm = gen_input(rho_cube_p, mol.spin, xc_type)
            else:
                rho_cube_p = [
                    pyscf.dft.numint.eval_rho(
                        mol, ao_cube, dm1_input[0], xctype="mGGA", with_lapl=False
                    ),
                    pyscf.dft.numint.eval_rho(
                        mol, ao_cube, dm1_input[1], xctype="mGGA", with_lapl=False
                    ),
                ]
                rho_cube_p_norm = gen_input(rho_cube_p, mol.spin, xc_type)
            return np.reshape(rho_cube_p_norm, (LEN, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

        with parallel_config(
            backend="loky",
            n_jobs=int(os.environ.get("OMP_NUM_THREADS", 6)),
            inner_max_num_threads=1,
        ):
            rho_cube = Parallel()(
                delayed(gen_cube_rho_p)(p) for p in range(len(self.coords))
            )

        return np.array(rho_cube)

    def get_center_rho(self, rho_cube):
        return (
            rho_cube[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + rho_cube[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        )

    def get_center_density(self, den_cube):
        return den_cube[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
