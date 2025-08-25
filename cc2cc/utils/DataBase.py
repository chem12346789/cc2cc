"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

import os
from itertools import product

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cc2cc.utils.env_var import DATA_PATH
from cc2cc.utils.mol import gen_mole


class BasicDataset(Dataset):
    """
    Documentation for a class.
    """

    def __init__(self, name_list, mol_info_dict, load_data):
        super(BasicDataset, self).__init__()
        self.data = {}
        self.name_list = []

        for name in name_list:
            num_data_used, data_dict = load_data(mol_info_dict[name], name)
            if num_data_used != 0:
                self.data[name] = data_dict
                self.name_list.append(name)
            # Add more copies of the atomic data to balance the dataset.
            # This is useful when we need to have more data for single-atom systems.
            if num_data_used == 1:
                self.name_list.extend([name] * 9)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        return self.data[self.name_list[idx]]

    def get_from_name(self, name):
        """
        Get the data from the name.
        """
        if name in self.data:
            return self.data[name]
        else:
            raise KeyError(f"Data for {name} not found.")


class DataBase:
    """Documentation for a class."""

    def __init__(
        self,
        molecule_list,
        args,
        shuffle=True,
    ):
        """
        Initialize the DataBase with a list of molecules and arguments.
        Args:
            molecule_list (list): List of molecule names to include in the database.
            args: Arguments containing various settings for the database.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.rho_input = args.rho_input
        if args.precision == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        self.train_atom = args.train_atom

        name_list = []
        error_molecule = []
        mol_info_dict = {}
        self.atomic_name_dict = {}

        training_cycle_list = [""]
        if args.training_cycle > 0:
            training_cycle_list.extend(
                [f"_scf_{i}" for i in range(1, args.training_cycle + 1)]
            )

        for (
            name_mol,
            extend_atom,
            extend_xyz,
            distance,
            training_cycle_iteration,
        ) in product(
            molecule_list,
            args.extend_atom,
            args.extend_xyz,
            args.distance_list,
            training_cycle_list,
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
                name = f"{name}{training_cycle_iteration}"

                path_name_ = DATA_PATH / f"data_{name}.npz"
                if not (path_name_).exists():
                    print(f"No file: {name:>40}", flush=True)
                    error_molecule.append(name)
                    print(f"Error molecule: {error_molecule}")
                    continue

                name_list.append(name)
                if mol.natm == 1 and mol.charge == 0:
                    self.atomic_name_dict[mol.atom_pure_symbol(0)] = name
                    print(f"{mol.elements} use {name}", flush=True)

                mol_info_dict[name] = {
                    "natm": mol.natm,
                    "elements": mol.elements,
                    "charge": mol.charge,
                    "spin": mol.spin,
                }

            except ValueError as e:
                print(f"Error generating molecule {name}: {e}", flush=True)

        self.dataset = BasicDataset(name_list, mol_info_dict, self.load_data)
        if args.distributed:
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, shuffle=shuffle
            )
            shuffle = False
        else:
            self.sampler = None
        self.data_gpu = DataLoader(
            self.dataset,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=int(os.environ.get("OMP_NUM_THREADS", 0)),
            pin_memory=True,
            sampler=self.sampler,
        )

    def __len__(self):
        return len(self.dataset.name_list)

    def process_batch(self, batch, device="cuda"):
        """
        Load the batch data to the GPU.
        Note all data is in the list ([data]), so we need to access the first element.
        """
        batch_gpu = {}
        for key, val in batch.items():
            if key in ["input", "weight", "output"]:
                batch_gpu[key] = val[0].to(
                    device=device,
                    non_blocking=True,
                )
            else:
                batch_gpu[key] = val[0]
        return batch_gpu

    def process_batch_dataset(self, batch, device="cuda"):
        """
        Load the batch data to the GPU.
        Note all data is in the list ([data]), so we need to access the first element.
        """
        batch_gpu = {}
        for key, val in batch.items():
            if key in ["input", "weight", "output"]:
                batch_gpu[key] = val.to(
                    device=device,
                    non_blocking=True,
                )
            else:
                batch_gpu[key] = val
        return batch_gpu

    def load_data(self, mol_info, name):
        """
        Load the data.
        """
        raise NotImplementedError(
            "The load_data method should be implemented in the subclass."
        )
