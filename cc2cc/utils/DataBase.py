"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

from itertools import product
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from cc2cc.utils.env_var import DATA_PATH, CUBE_SIZE
from cc2cc.utils.mol import gen_mole, AU2KCALMOL


class BasicDataset:
    """
    Documentation for a class.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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

    def __init__(self, molecule_list, args):
        self.train_atom = args.train_atom
        self.if_load_to_gpu_once = args.if_load_to_gpu_once
        print(f"Load to GPU once: {self.if_load_to_gpu_once}")

        if args.precision == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32

        self.data = []
        self.data_weight = {}  # weight for each data, some atom may have more weight
        self.data_weight_mol = {}

        self.name_list = []
        error_molecule = []
        self.atomic_name_dict = {}

        for (
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
        ) in product(
            molecule_list,
            args.extend_atom,
            args.extend_xyz,
            args.distance_list,
        ):
            name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

            try:
                mol = gen_mole(
                    name_mol,
                    extend_atom,
                    extend_xyz,
                    distance,
                    args.basis,
                    args.if_basis_str,
                    args.dataset,
                    verbose=-1,
                )
            except ValueError as e:
                print(f"SKIP: {name}")
                print(e)
                error_molecule.append(name)
                print(f"Error molecule: {error_molecule}")
                continue
            finally:
                if args.n_rad is not None and args.n_ang is not None:
                    name = f"{name}_{args.n_rad}_{args.n_ang}"
                else:
                    name = f"{name}_default"

                path_name_ = DATA_PATH / f"data_{name}.npz"
                if not (path_name_).exists():
                    print(f"No file: {name:>40}", flush=True)
                    error_molecule.append(name)
                    print(f"Error molecule: {error_molecule}")
                    continue

                num_data_used = self.load_data(mol, name)
                if num_data_used == 0:
                    error_molecule.append(name)
                    print(f"Error molecule: {error_molecule}")
                else:
                    self.name_list.append(name)

                if name_mol not in self.data_weight_mol:
                    self.data_weight_mol[name_mol] = num_data_used
                else:
                    self.data_weight_mol[name_mol] = max(
                        self.data_weight_mol[name_mol],
                        num_data_used,
                    )

                if mol.natm == 1 and mol.charge == 0:
                    self.atomic_name_dict[mol.atom_pure_symbol(0)] = name
                    print(f"{mol.atom_pure_symbol(0)} use {name}", flush=True)

        name_extend = []
        for name in self.name_list:
            name_mol = name.split(f"_{args.basis}_")[0]
            # if self.data_weight_mol[name_mol] == 1:
            #     name_extend.extend([name] * 9)
            self.data_weight[name] = 1 / self.data_weight_mol[name_mol]
        self.name_list.extend(name_extend)
        del self.data_weight_mol
        print(self.data_weight)

        self.data_gpu = BasicDataset(self.data)
        self.data_gpu = self.load_to_gpu()

    def load_data(self, mol, name):
        """
        Load the data.
        """
        data = np.load(DATA_PATH / f"data_{name}.npz")

        input_mat = data["rho_cube"]
        weight_mat = data["weights"]
        output_mat = data["exc_cc_grids"]

        # print(f"Total energy real: {AU2KCALMOL * data['error_energy']}")
        # print(f"Total energy: {AU2KCALMOL * np.sum(output_mat * weight_mat)}")
        if (
            AU2KCALMOL * abs(data["error_energy"] - np.sum(output_mat * weight_mat))
            > 0.2 * mol.natm
        ):
            print(f"Error energy is too large: {name:>40}", flush=True)
            return 0

        input_ = []
        weight_ = []
        output_ = []
        atomic_systems = []
        atomic_stoichiometry = []

        num_data_used = 0
        total_ene_used = 0
        data_length = len(input_mat) // mol.natm
        for i_atom in range(mol.natm):
            atom_name = mol.atom_pure_symbol(i_atom)
            if self.train_atom not in ["all", "All", "ALL"]:
                if atom_name != self.train_atom:
                    print(
                        f"SKIP: {name:>40} {atom_name:>3}",
                        flush=True,
                    )
                    continue

            # print(f"Load: {name:>40} {atom_name:>3}", flush=True)
            if atom_name not in atomic_systems:
                atomic_systems.append(atom_name)
                atomic_stoichiometry.append(1)
            else:
                atomic_stoichiometry[atomic_systems.index(atom_name)] += 1

            num_data_used += 1
            slice_ = slice(data_length * i_atom, data_length * (i_atom + 1))
            input_.append(input_mat[slice_, :, :, :, :])
            weight_.append(weight_mat[slice_])
            output_.append(output_mat[slice_])
            total_ene_used += np.sum(output_mat[slice_] * weight_mat[slice_])
        input_ = np.array(input_).reshape((-1, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        weight_ = np.array(weight_).reshape((-1, 1))
        output_ = np.array(output_).reshape((-1, 1))

        if num_data_used == 0:
            return 0

        print(f"Total energy used: {AU2KCALMOL * total_ene_used}")
        print(f"Total data used for {name}: {num_data_used}", flush=True)
        print(
            f"Atomic systems: {atomic_systems}, Stoichiometry: {atomic_stoichiometry}",
            flush=True,
        )
        self.data.append(
            {
                "input": np.array(input_),
                "weight": np.array(weight_),
                "output": np.array(output_),
                "name": name,
                "atomic_systems": atomic_systems,
                "atomic_stoichiometry": atomic_stoichiometry,
            }
        )

        return num_data_used

    def process(self, data):
        """
        Load the data to the GPU.
        """
        return data.to(device="cuda", dtype=self.dtype)

    def process_batch(self, batch):
        """
        Load the batch data to the GPU.
        """
        batch_gpu = {}
        for key, val in batch.items():
            if key in ["input", "weight", "output"]:
                batch_gpu[key] = self.process(val[0])
            elif key in ["name"]:
                batch_gpu[key] = val[0]
            else:
                # For other keys, we just keep them as they are
                batch_gpu[key] = val[0]
        return batch_gpu

    def load_to_gpu(self):
        """
        Load the whole data to the gpu.
        """
        dataloader = DataLoader(
            self.data_gpu,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
        )

        dataloader_gpu = {}
        for batch in dataloader:
            if self.if_load_to_gpu_once:
                dataloader_gpu[batch["name"][0]] = self.process_batch(batch)
            else:
                dataloader_gpu[batch["name"][0]] = batch
        return dataloader_gpu

    def shuffle(self):
        """
        Shuffle the data.
        """
        random.shuffle(self.name_list)
