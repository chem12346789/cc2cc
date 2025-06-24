"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

import os
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cc2cc.utils.env_var import DATA_PATH, CUBE_SIZE
from cc2cc.utils.mol import gen_mole, AU2KCALMOL


class BasicDataset(Dataset):
    """
    Documentation for a class.
    """

    def __init__(self, name_list, mol_info_dict, load_data):
        super(BasicDataset, self).__init__()
        self.data = {}
        self.name_list = []
        self.data_weight = {}

        for name in name_list:
            num_data_used, data_dict = load_data(mol_info_dict[name], name)
            self.data_weight[name] = 1 / num_data_used
            if num_data_used > 0:
                self.data[name] = data_dict
                self.name_list.append(name)
            # Add more copies of the atomic data to balance the dataset.
            # This is useful when we need to have more data for single-atom systems.
            if num_data_used == 1:
                self.name_list.extend([name] * 9)
        print(self.data_weight)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        return self.data[self.name_list[idx]]


class DataBase:
    """Documentation for a class."""

    def __init__(self, molecule_list, args):
        self.rho_dft = args.rho_dft
        if args.precision == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        self.train_atom = args.train_atom
        self.if_load_to_gpu_once = args.if_load_to_gpu_once
        print(f"Load to GPU once: {self.if_load_to_gpu_once}")

        self.name_list = []
        error_molecule = []
        self.atomic_name_dict = {}
        self.mol_info_dict = {}

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

                self.name_list.append(name)
                if mol.natm == 1 and mol.charge == 0:
                    self.atomic_name_dict[mol.atom_pure_symbol(0)] = name
                    print(f"{mol.elements} use {name}", flush=True)

                self.mol_info_dict[name] = {
                    "natm": mol.natm,
                    "elements": mol.elements,
                    "charge": mol.charge,
                    "spin": mol.spin,
                }

            except ValueError as e:
                print(f"Error generating molecule {name}: {e}", flush=True)

        self.dataset = BasicDataset(self.name_list, self.mol_info_dict, self.load_data)
        self.data_gpu = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=1,
            num_workers=int(os.environ.get("NUMBER_OF_THREADS", 1)),
            pin_memory=True,
        )

    def process_batch(self, batch):
        """
        Load the batch data to the GPU.
        Note all data is in the list ([data]), so we need to access the first element.
        """
        batch_gpu = {}
        for key, val in batch.items():
            # key in ["input", "weight", "output"] and val is not in GPU
            if key in ["input", "weight", "output"] and not self.if_load_to_gpu_once:
                batch_gpu[key] = val[0].to(device="cuda", dtype=self.dtype)
                continue
            batch_gpu[key] = val[0]
        return batch_gpu

    def load_data(self, mol_info, name):
        """
        Load the data.
        """
        print("", flush=True)
        data = np.load(DATA_PATH / f"data_{name}.npz")

        if self.rho_dft:
            input_mat = data["rho_cube_dft"]
        else:
            input_mat = data["rho_cube_cc"]
        weight_mat = data["weights"]
        output_mat = data["exc_cc_grids"]

        # print(f"Total energy real: {AU2KCALMOL * data['error_energy']}")
        # print(f"Total energy: {AU2KCALMOL * np.sum(output_mat * weight_mat)}")
        if (
            AU2KCALMOL * abs(data["error_energy"] - np.sum(output_mat * weight_mat))
            > 0.2 * mol_info["natm"]
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
        data_length = len(input_mat) // mol_info["natm"]
        for i_atom in range(mol_info["natm"]):
            atom_name = mol_info["elements"][i_atom]
            if self.train_atom not in ["all", "All", "ALL"]:
                if atom_name != self.train_atom:
                    print(
                        f"SKIP: {name:>40} {atom_name:>3}",
                        flush=True,
                    )
                    continue

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

        input_ = torch.tensor(input_, dtype=self.dtype)
        weight_ = torch.tensor(weight_, dtype=self.dtype)
        output_ = torch.tensor(output_, dtype=self.dtype)

        if self.if_load_to_gpu_once:
            input_ = input_.to(device="cuda", dtype=self.dtype, non_blocking=True)
            weight_ = weight_.to(device="cuda", dtype=self.dtype, non_blocking=True)
            output_ = output_.to(device="cuda", dtype=self.dtype, non_blocking=True)

        data_dict = {
            "input": input_,
            "weight": weight_,
            "output": output_,
            "name": name,
            "atomic_systems": atomic_systems,
            "atomic_stoichiometry": atomic_stoichiometry,
        }

        return num_data_used, data_dict
