"""@package docstring
Documentation for this module.
 
More details.
"""

import ctypes

import numpy as np
import pyscf
from pyscf import dft, gto, lib

from cc2cc.utils.env_var import LEVEL, PERIOD

libdft = lib.load_library("libdft")
AU2ANG = 0.52917721067

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


def get_inertia_moment(
    rho_atom_matrix_i_atom,
    weight_matrix_i_atom,
    corr_matrix_rotated_i_atom,
):
    inertia_electron = np.zeros((3, 3), dtype=np.float64)
    weighted_density = rho_atom_matrix_i_atom * weight_matrix_i_atom

    # Calculate diagonal elements
    for i in range(3):
        inertia_electron[i, i] = np.sum(
            weighted_density * corr_matrix_rotated_i_atom[i, :, :] ** 2
        )

    # Calculate off-diagonal elements
    for i in range(3):
        for j in range(i + 1, 3):
            inertia_electron[i, j] = -np.sum(
                weighted_density
                * corr_matrix_rotated_i_atom[i, :, :]
                * corr_matrix_rotated_i_atom[j, :, :]
            )
            inertia_electron[j, i] = inertia_electron[i, j]

    dipole_electron = np.array(
        [
            np.sum(weighted_density * corr_matrix_rotated_i_atom[i, :, :])
            for i in range(3)
        ]
    )

    # print(f"inertia_electron: {inertia_electron}")
    eig_val, eig_vec = np.linalg.eigh(inertia_electron)
    for i in range(3):
        if eig_vec[:, i] @ dipole_electron < 0:
            eig_vec[:, i] *= -1

    eig_vec[:, :] = eig_vec[:, np.argsort(eig_val)]
    eig_val = eig_val[np.argsort(eig_val)]
    return eig_val, eig_vec


def gen_input(rho, spin, xc_type):
    """
    Generate the input for the model.
    """
    if spin == 0:
        rho0 = rho[0]
        rho_lda = rho[0]
    else:
        rho0 = rho[0][0] + rho[1][0]
        rho_lda = [rho[0][0], rho[1][0]]

    lda_grids = pyscf.dft.libxc.eval_xc("LDA,", rho_lda, spin)[0]
    vwn_grids = pyscf.dft.libxc.eval_xc(",VWN3", rho_lda, spin)[0]
    b88_grids = pyscf.dft.libxc.eval_xc("B88,", rho, spin)[0]
    lyp_grids = pyscf.dft.libxc.eval_xc(",LYP", rho, spin)[0]

    rho_out = np.array(
        [
            lda_grids * rho0,
            vwn_grids * rho0,
            b88_grids * rho0,
            lyp_grids * rho0,
            rho0,
        ]
    )

    return rho_out


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
                raise ValueError(f"Atom {symb} not found in atom_grid")
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

        self.index_2d = None

    def vector_to_matrix(self, vector: np.ndarray):
        """
        Documentation for a method.
        """
        matrix = np.zeros((self.natm, self.n_rad, self.n_ang))
        index_range = np.ndindex(self.natm, self.n_rad, self.n_ang)
        for i, j, k in index_range:
            matrix[i, j, k] = vector[self.index_2d[i, j, k]]
        return matrix

    def matrix_to_vector(self, matrix: np.ndarray):
        """
        Documentation for a method.
        """
        vector = np.zeros(self.natm * self.n_rad * self.n_ang)
        index_range = np.ndindex(self.natm, self.n_rad, self.n_ang)
        for i, j, k in index_range:
            vector[self.index_2d[i, j, k]] = matrix[i, j, k]
        return vector

    def gen_grids(self, mol, dm1_input):
        """
        Generate the cube coordinates for the given molecule.

        Args:
            mol: An instance of :class:`Mole'.
            dm1_input: Density matrix, 2D array with shape (nao, nao). The orientation of the cube is determined by the eigenvectors of the Hessian matrix(secondary derivation of the density).
        """
        # return
        self.index_2d = np.arange(len(self.coords)).reshape(
            self.natm, self.n_ang, self.n_rad
        )
        self.index_2d = np.transpose(self.index_2d, axes=[0, 2, 1])

        ao_value = pyscf.dft.numint.eval_ao(mol, self.coords)
        rho_atom = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_input, xctype="LDA")
        rho_atom_matrix = self.vector_to_matrix(rho_atom * self.weights)
        weight_matrix = self.vector_to_matrix(self.weights)
        corr_x_matrix = self.vector_to_matrix(self.coords[:, 0])
        corr_y_matrix = self.vector_to_matrix(self.coords[:, 1])
        corr_z_matrix = self.vector_to_matrix(self.coords[:, 2])
        corr_matrix_rotated = np.zeros((self.natm, 3, self.n_rad, self.n_ang))
        # 4 3 75 302

        for i_atom in range(self.natm):
            corr_matrix_rotated[i_atom] = np.array(
                [
                    corr_x_matrix[i_atom, :, :] - self.coord_list[i_atom][0] / AU2ANG,
                    corr_y_matrix[i_atom, :, :] - self.coord_list[i_atom][1] / AU2ANG,
                    corr_z_matrix[i_atom, :, :] - self.coord_list[i_atom][2] / AU2ANG,
                ]
            )
            eig_val, eig_vec = get_inertia_moment(
                rho_atom_matrix[i_atom],
                weight_matrix[i_atom],
                corr_matrix_rotated[i_atom],
            )

            if np.linalg.norm(eig_val - np.mean(eig_val)) > 1e-8:
                # print(f"Warning: eig_val is not zero! {eig_val} {eig_vec}")
                rad_orientation = np.array(
                    [
                        corr_matrix_rotated[i_atom, 0, -1, :]
                        - corr_matrix_rotated[i_atom, 0, 0, :],
                        corr_matrix_rotated[i_atom, 1, -1, :]
                        - corr_matrix_rotated[i_atom, 1, 0, :],
                        corr_matrix_rotated[i_atom, 2, -1, :]
                        - corr_matrix_rotated[i_atom, 2, 0, :],
                    ]
                )
                rad_dot_eig = np.einsum("ij,ik->jk", rad_orientation, eig_vec)
                rad_dot_eig_sort = np.lexsort(
                    (rad_dot_eig[:, 2], rad_dot_eig[:, 1], rad_dot_eig[:, 0])
                )
                self.index_2d[i_atom, :, :] = np.take(
                    self.index_2d[i_atom, :, :], rad_dot_eig_sort, axis=1
                )

    def gen_grids_matrix(self, mol, dm1_input, reset=False, xc_type="GGA"):
        """
        Generate the cube coordinates for the given molecule.
        """
        if self.index_2d is None:
            print("Warning: generate index!")
            self.gen_grids(mol, dm1_input)
        elif reset:
            print("Warning: regenerate index!")
            self.gen_grids(mol, dm1_input)
        else:
            print("Warning: Use the existing coor_cube!")

        ao_cube = pyscf.dft.numint.eval_ao(mol, self.coords, deriv=1)
        if mol.spin == 0:
            rho = pyscf.dft.numint.eval_rho(mol, ao_cube, dm1_input, xctype=xc_type)
            rho_norm = gen_input(rho, 0, xc_type)
            rho_norm_matrix = np.array(
                [self.vector_to_matrix(rho_norm[i]) for i in range(len(rho_norm))]
            )
        else:
            rho = [
                pyscf.dft.numint.eval_rho(mol, ao_cube, dm1_input[0], xctype=xc_type),
                pyscf.dft.numint.eval_rho(mol, ao_cube, dm1_input[1], xctype=xc_type),
            ]
            rho_norm = gen_input(rho, 1, xc_type)
            rho_norm_matrix = np.array(
                [self.vector_to_matrix(rho_norm[i]) for i in range(len(rho_norm))]
            )
        return np.transpose(rho_norm_matrix, axes=[1, 0, 2, 3])
