"""
Get the input for the model.
"""

from itertools import product

import numpy as np
import pyscf

from dft2cc.utils.Grids import Grid
from dft2cc.utils.DataBase import process_input
from dft2cc.utils.env_var import (
    STRUCTURE,
    CUBE_USE,
    CUBE_LEN,
    CUBE_USE_MIDDLE,
)


def get_input_mat(
    dft: pyscf.dft.rks.RKS,
    grids: Grid,
    dms: np.ndarray = None,
):
    """
    Get the input matrix for the model.
    Input:
    dft: the dft instance, RKS/UKS object; See https://pyscf.org/_modules/pyscf/dft/rks.html
    grids: the grids instance, Grids object; See https://pyscf.org/_modules/pyscf/dft/numint.html and the modified version in dft2cc/utils/Grids.py
    """
    if isinstance(dft, pyscf.dft.rks.RKS):
        if "3d" in STRUCTURE:
            grids = Grid(dft.mol, level=1, period=2)

            rho_cube = np.zeros((len(grids.coords), 4, CUBE_USE, CUBE_USE, CUBE_USE))
            for p, p_coords in enumerate(grids.coords):
                if p * 10 % len(grids.coords) == 0:
                    print(f"Progress: {(p*100)/len(grids.coords):.1f}%", flush=True)

                coords_cube = np.zeros((CUBE_USE, CUBE_USE, CUBE_USE, 3))
                for i, j, k in product(range(CUBE_USE), repeat=3):
                    coords_cube[i, j, k] = p_coords + [
                        (i - CUBE_USE_MIDDLE) * CUBE_LEN,
                        (j - CUBE_USE_MIDDLE) * CUBE_LEN,
                        (k - CUBE_USE_MIDDLE) * CUBE_LEN,
                    ]
                coords_cube = coords_cube.reshape(-1, 3)

                ao_cube = pyscf.dft.numint.eval_ao(dft.mol, coords_cube, deriv=1)
                rho_cube_p = pyscf.dft.numint.eval_rho(
                    dft.mol, ao_cube, dms, xctype="GGA"
                )
                rho_cube[p] = rho_cube_p.reshape(4, CUBE_USE, CUBE_USE, CUBE_USE)
            return rho_cube
        if "unet" in STRUCTURE:
            ao_value = pyscf.dft.numint.eval_ao(dft.mol, grids.coords, deriv=1)
            scf_r_3 = pyscf.dft.numint.eval_rho(dft.mol, ao_value, dms, xctype="GGA")
            input_mat = process_input(scf_r_3, grids)
            input_mat = np.transpose(input_mat, (1, 0, 2, 3))
            input_mat = input_mat[:, [0], :, :]
            return input_mat
