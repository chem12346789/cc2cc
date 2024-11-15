from pathlib import Path
from itertools import product

import numpy as np

from cc2cc.utils import (
    AU2KCALMOL,
    DATA_PATH,
    CUBE_USE,
    CUBE_MIDDLE,
    CUBE_USE_MIDDLE,
    ARRAY_USE_MIDDLE,
    LEVEL,
    PERIOD,
)

def load_data(
    molecular_list,
    extend_atom,
    extend_xyz,
    distance_list,
    basis="cc-pVDZ",
    view_keys=[],
):
    """
    Load the data.
    """
    input_dict = {}
    output_dict = {}
    weights_dict = {}
    coords_dict = {}
    keys_list = []
    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        molecular_list,
        extend_atom,
        extend_xyz,
        distance_list,
    ):
        name = f"{name_mol}_{basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        data_path = Path(f"{DATA_PATH}") / f"data_{name}_{LEVEL}_{PERIOD}.npz"
        if not (data_path).exists():
            print(f"No file: {data_path}")
            continue
        else:
            print(f"Load the data: {data_path}")

        data = np.load(data_path)

        if view_keys:
            output_ = []
            for key in view_keys:
                if "+" in key:
                    data_ = []
                    for key_i in key.split("+"):
                        data_.append(data[key_i])
                    output_.append(np.sum(data_, axis=0))
                else:
                    output_.append(data[key])
            output_ = np.array(output_).T
        else:
            if "exc_over_dm_mrks_grids" in data.files:
                output_ = data["exc_over_dm_mrks_grids"]
            else:
                output_ = data["exc_over_dm_cc_grids"]
        print(output_.shape)

        weights_ = data["weights"]
        coords_cube = data["coor_cube"]
        coords_cube = coords_cube[
            :,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            :,
        ]
        input_ = data["rho_cube"]
        input_ = input_[
            :,
            :,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
        ]

        input_[:, 0, :, :, :] = input_[:, 0, :, :, :]
        input_[:, 1, :, :, :] = input_[:, 1, :, :, :] ** (1 / 2)
        input_ = input_.reshape(-1, (CUBE_USE) ** 3 * 2)
        # input_ = np.transpose(data["rho_inv_4_norm"], (1, 0))

        input_dict[name] = input_
        output_dict[name] = output_
        weights_dict[name] = weights_
        print(f"max input: {np.max(input_)}, min input: {np.min(input_)}")

        coords_cube = coords_cube.reshape(-1, (CUBE_USE) ** 3, 3)
        coords_dict[name] = coords_cube

        keys_list.append(name)
        if len(np.shape(output_dict[name])) == 1:
            print(
                AU2KCALMOL
                * np.sum(
                    np.abs(
                        input_dict[name][:, ARRAY_USE_MIDDLE]
                        * output_dict[name]
                        * weights_dict[name]
                    )
                ),
                AU2KCALMOL
                * np.sum(
                    input_dict[name][:, ARRAY_USE_MIDDLE]
                    * output_dict[name]
                    * weights_dict[name]
                ),
            )
    return input_dict, output_dict, weights_dict, coords_dict, keys_list
