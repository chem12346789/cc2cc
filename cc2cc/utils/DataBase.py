from pathlib import Path
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from cc2cc.utils.env_var import DATA_PATH, CUBE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL


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

            path_name_ = DATA_PATH / f"data_{name}.npz"
            if not (path_name_).exists():
                print(f"No file: {path_name_.as_posix():>40}", flush=True)
                continue
            print(f"Load: {name:>40}", flush=True)
            self.name_list.append(name)
            self.load_data(name)

    def load_data(self, name):
        """
        Load the data.
        """
        data = np.load(DATA_PATH / f"data_{name}.npz")

        input_mat = data["rho_cube"]
        weight_mat = data["weights"]
        output_mat = data["exc_cc_grids"]

        print(AU2KCALMOL * data["error_energy"])
        print(AU2KCALMOL * np.sum(output_mat * weight_mat))
        print(f"{np.min(input_mat)}, {np.max(input_mat)}")
        print(f"{np.min(output_mat)}, {np.max(output_mat)}")

        input_ = {}
        weight_ = {}
        output_ = {}

        for i_coord in range(len(input_mat)):
            input_[i_coord] = input_mat[i_coord, :, :, :, :]
            weight_[i_coord] = weight_mat[[i_coord]]
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
