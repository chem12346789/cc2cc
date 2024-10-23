"""@package docstring
Documentation for this module.
 
More details.
"""

import ctypes

import numpy as np
from pyscf import dft

import pyscf
from pyscf import gto
from pyscf import lib

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

    def __init__(self, mol, level=1, period=2):
        super().__init__(mol)
        self.n_rad, self.n_ang = (
            RAD_GRIDS[level, period],
            LEBEDEV_ORDER[ANG_ORDER[level, period]],
        )
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

        self.index_2d = np.arange(len(self.coords)).reshape(
            self.natm, self.n_ang, self.n_rad
        )
        self.index_2d = np.transpose(self.index_2d, axes=[0, 2, 1])

    def vector_to_matrix(self, vector: np.ndarray):
        """
        Documentation for a method.
        """
        matrix = np.empty((self.natm, self.n_rad, self.n_ang), dtype=vector.dtype)
        index_range = np.ndindex(self.natm, self.n_rad, self.n_ang)
        for i, j, k in index_range:
            matrix[i, j, k] = vector[self.index_2d[i, j, k]]
        return matrix

    def matrix_to_vector(self, matrix: np.ndarray):
        """
        Documentation for a method.
        """
        vector = np.empty(self.natm * self.n_rad * self.n_ang, dtype=matrix.dtype)
        index_range = np.ndindex(self.natm, self.n_rad, self.n_ang)
        for i, j, k in index_range:
            vector[self.index_2d[i, j, k]] = matrix[i, j, k]
        return vector

    def matrix_to_vector_atom(self, matrix: np.ndarray, atom_number: int):
        """
        Documentation for a method.
        """
        atom_x = np.zeros(self.n_rad * self.n_ang)
        atom_y = np.zeros(self.n_rad * self.n_ang)
        atom_z = np.zeros(self.n_rad * self.n_ang)
        vector = np.empty(self.n_rad * self.n_ang, dtype=matrix.dtype)
        index_range = np.ndindex(self.n_rad, self.n_ang)
        for i, (j, k) in enumerate(index_range):
            vector[i] = matrix[atom_number, j, k]
            atom_x[i] = self.coords[:, 0][self.index_2d[atom_number, j, k]]
            atom_y[i] = self.coords[:, 1][self.index_2d[atom_number, j, k]]
            atom_z[i] = self.coords[:, 2][self.index_2d[atom_number, j, k]]
        return atom_x, atom_y, atom_z, vector
