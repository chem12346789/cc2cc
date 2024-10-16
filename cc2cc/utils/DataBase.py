from pathlib import Path
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from cc2cc.utils.env_var import DATA_PATH, STRUCTURE, CUBE_MIDDLE, CUBE_USE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL


def process_input(data, grids):
    """
    process the input
    """
    data_grids_norm = np.zeros((4, len(grids.coord_list), grids.n_rad, grids.n_ang))
    for oxyz in range(4):
        if oxyz == 0:
            data_grids_norm[oxyz, :, :, :] = grids.vector_to_matrix(data[oxyz, :])
        else:
            data_grids_norm[oxyz, :, :, :] = grids.vector_to_matrix(data[oxyz, :])
    return data_grids_norm


def process(data, dtype):
    """
    Load the whole data to the gpu.
    """
    if len(data.shape) == 4:
        return data.to(
            device="cuda",
            dtype=dtype,
            memory_format=torch.channels_last,
        )
    else:
        return data.to(
            device="cuda",
            dtype=dtype,
        )


class BasicDataset:
    """
    Documentation for a class.
    """

    def __init__(self, dict_batch, batch_size, dtype, dict_const=None):
        self.dict_batch = dict_batch
        self.dict_const = dict_const
        self.ids = list(dict_batch["input"].keys())
        self.batch_size = batch_size
        if dtype == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        dict_out = {}
        for key, val in self.dict_batch.items():
            dict_out[key] = val[idx]
        return dict_out

    def load_to_gpu(self):
        """
        Load the whole data to the device.
        """
        dataloader = DataLoader(
            self,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
        )

        dataloader_gpu = []
        for batch in dataloader:
            batch_gpu = {}
            for key, val in batch.items():
                batch_gpu[key] = process(val, self.dtype)
            if self.dict_const is not None:
                for key, val in self.dict_const.items():
                    batch_gpu[key] = torch.tensor(val, dtype=self.dtype).to(
                        device="cuda"
                    )
            dataloader_gpu.append(batch_gpu)
            if self.dict_const is not None:
                if len(dataloader_gpu) > 1:
                    raise ValueError("Only one batch is allowed.")
        return dataloader_gpu


def gen_logger(distance_list):
    """
    Function to distance list and generate logger
    """
    if len(distance_list) == 3:
        distance_l = np.linspace(
            distance_list[0], distance_list[1], int(distance_list[2])
        )
    else:
        distance_l = distance_list
    return distance_l


class DataBase:
    """Documentation for a class."""

    def __init__(
        self,
        molecular_list,
        extend_atom,
        extend_xyz,
        distance_list,
        basis,
        batch_size,
        device,
        dtype,
    ):
        self.molecular_list = molecular_list
        self.extend_atom = extend_atom
        self.extend_xyz = extend_xyz
        self.distance_list = distance_list
        self.basis = basis
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.data = {}
        self.data_gpu = {}
        self.ene = {}
        self.shape = {}

        self.name_list = []
        self.rng = np.random.default_rng()

        for (
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
        ) in product(
            self.molecular_list,
            self.extend_atom,
            self.extend_xyz,
            self.distance_list,
        ):
            name = f"{name_mol}_{self.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

            if "openshell" in name:
                for i_spin in range(2):
                    name_ = f"{name}_{i_spin}"
                    if not (Path(f"{DATA_PATH}") / f"data_{name_}.npz").exists():
                        print(f"No file: {name_:>40}", flush=True)
                        continue
                    print(f"Load: {name_:>40}", flush=True)
                    self.name_list.append(f"{name_}")
                    self.load_data(name_)
            else:
                if not (Path(f"{DATA_PATH}") / f"data_{name}.npz").exists():
                    print(f"No file: {name:>40}", flush=True)
                    continue
                print(f"Load: {name:>40}", flush=True)
                self.name_list.append(name)
                self.load_data(name)

    def load_data(self, name):
        """
        Load the data.
        """
        data = np.load(Path(f"{DATA_PATH}") / f"data_{name}.npz")
        print(AU2KCALMOL * data["error_energy"])

        if "unet" in STRUCTURE:
            input_mat = data["rho_inv_4_norm_matrix"]
            output_mat = data["exc_over_dm_cc_grids_matrix"]
            weights_mat = data["weights_matrix"]

            input_ = {}
            weight_ = {}
            output_ = {}

            for i_atom in range(output_mat.shape[0]):
                input_[i_atom] = input_mat[[0], i_atom, :, :]
                weight_[i_atom] = weights_mat[[i_atom], :, :]
                output_[i_atom] = output_mat[[i_atom], :, :]

            self.data_gpu[name] = BasicDataset(
                {
                    "input": input_,
                    "output": output_,
                    "weight": weight_,
                },
                self.batch_size,
                self.dtype,
            ).load_to_gpu()
        else:
            if "3d" in STRUCTURE:
                input_mat = data["rho_cube"]
            else:
                input_mat = data["rho_inv_4_norm"]
            output_mat = data["exc_over_dm_cc_grids"]
            weights_mat = data["weights"]

            input_ = {}
            weight_ = {}
            output_ = {}

            for i_coord in range(len(weights_mat)):
                if "3d" in STRUCTURE:
                    input_[i_coord] = input_mat[
                        i_coord,
                        :,
                        CUBE_MIDDLE
                        - CUBE_USE_MIDDLE : CUBE_MIDDLE
                        + CUBE_USE_MIDDLE
                        + 1,
                        CUBE_MIDDLE
                        - CUBE_USE_MIDDLE : CUBE_MIDDLE
                        + CUBE_USE_MIDDLE
                        + 1,
                        CUBE_MIDDLE
                        - CUBE_USE_MIDDLE : CUBE_MIDDLE
                        + CUBE_USE_MIDDLE
                        + 1,
                    ]
                else:
                    input_[i_coord] = input_mat[:, i_coord]
                weight_[i_coord] = weights_mat[[i_coord]]
                output_[i_coord] = output_mat[[i_coord]]

            self.data_gpu[name] = BasicDataset(
                {
                    "input": input_,
                    "weight": weight_,
                    "output": output_,
                },
                self.batch_size,
                self.dtype,
            ).load_to_gpu()
