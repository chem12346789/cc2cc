"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

import os
from itertools import product
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cc2cc.utils.env_var import DATA_PATH
from cc2cc.utils.mol import gen_mole


PRINT_DEBUG = False


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
                self.name_list.extend([name] * 19)

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
        if_eval=False,
        atomic_name_dict=None,
        atomic_energy_dict=None,
    ):
        """
        Initialize the DataBase with a list of molecules and arguments.
        Args:
            molecule_list (list): List of molecule names to include in the database.
            args: Arguments containing various settings for the database.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.args = args
        if args.precision == "float64":
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        self.if_eval = if_eval
        self.array_key = ["input", "weight", "output", "grad2force"]

        if args.loss_ene == "L1Loss":
            self.loss_ene = torch.nn.L1Loss(reduction="sum")
            self.loss_ene_abs = torch.nn.L1Loss(reduction="sum")
            self.loss_ene_atomic = torch.nn.L1Loss(reduction="sum")
            self.loss_grad = torch.nn.L1Loss(reduction="sum")
        elif args.loss_ene == "MSELoss":
            self.loss_ene = torch.nn.MSELoss(reduction="sum")
            self.loss_ene_abs = torch.nn.MSELoss(reduction="sum")
            self.loss_ene_atomic = torch.nn.MSELoss(reduction="sum")
            self.loss_grad = torch.nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unknown loss function {args.loss_ene}")

        name_list = []
        error_molecule = []
        mol_info_dict = {}
        if atomic_name_dict is None:
            self.atomic_name_dict = {}
        else:
            self.atomic_name_dict = atomic_name_dict
        if atomic_energy_dict is None:
            self.atomic_energy_dict = {}
        else:
            self.atomic_energy_dict = atomic_energy_dict

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
                    ma_basis=True,
                    dataset_name=args.dataset,
                    verbose=-1,
                )
                name = f"{name}{training_cycle_iteration}"

                path_name_ = DATA_PATH / f"data_{name}.npz"
                if not (path_name_).exists():
                    print(f"No file: {name:>40}", flush=True)
                    error_molecule.append(name)
                    print(f"Error molecule: {error_molecule}")
                    continue

                name_list.append(name)
                if mol.natm == 1 and mol.charge == 0:
                    if atomic_name_dict is None:
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

        # move atomic_name_dict in the head of name_list.
        for iter_atom_name, (atom_key, atom_name) in enumerate(
            self.atomic_name_dict.items()
        ):
            if atom_name in name_list:
                name_list.remove(atom_name)
                name_list.insert(iter_atom_name, atom_name)
            else:
                print(
                    f"Warning: atomic {atom_name} as {atom_key} not in the dataset.",
                    flush=True,
                )
        print(name_list, flush=True)

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
            if key in self.array_key:
                if PRINT_DEBUG:
                    print(f"key : {key}, shape of val : {val.size()}", flush=True)
                batch_gpu[key] = val[0].to(device=device, non_blocking=True)
                if PRINT_DEBUG:
                    print(
                        f"After processing, key : {key}, type of val : {type(batch_gpu[key])}, shape of val : {batch_gpu[key].size()}",
                        flush=True,
                    )
            else:
                if PRINT_DEBUG:
                    print(
                        f"key : {key}, len of val : {len(val)}, val : {val}", flush=True
                    )
                if isinstance(val, list):
                    if len(np.shape(val)) == 1:
                        batch_gpu[key] = val[0]
                    elif len(np.shape(val)) == 2:
                        batch_gpu[key] = list(np.array(val)[:, 0])
                    elif len(np.shape(val)) == 3:
                        batch_gpu[key] = list(np.array(val)[:, :, 0])
                    elif len(np.shape(val)) == 4:
                        batch_gpu[key] = list(np.array(val)[:, :, :, 0])
                    else:
                        raise ValueError(
                            f"Unknown shape for key {val}: {np.shape(val)}"
                        )
                else:
                    batch_gpu[key] = val[0]
                if PRINT_DEBUG:
                    print(
                        f"After processing, key : {key}, type of val : {type(batch_gpu[key])}, val : {batch_gpu[key]}",
                        flush=True,
                    )
        return batch_gpu

    def process_batch_dataset(self, batch, device="cuda"):
        """
        Load the batch data to the GPU.
        """
        batch_gpu = {}
        for key, val in batch.items():
            if key in self.array_key:
                batch_gpu[key] = val.to(device=device, non_blocking=True)
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
