"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

import os
from itertools import product
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cc2cc.utils.mol import gen_mole
from cc2cc.utils.env_var import DATA_PATH
from cc2cc.utils.mol import AU2KCALMOL


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
                append_number = 20 // int(data_dict["data_weight"]) - 1
                self.name_list.extend([name] * append_number)

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
        process_input=lambda x: x,
        process_grad2force=lambda x: x,
        verbose=False,
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
        self.process_input = process_input
        self.process_grad2force = process_grad2force
        self.verbose = verbose
        self.array_key = ["input", "weight", "output", "grad2force"]

        if args.normal_loss_ene == "L1Loss":
            self.loss_ene = lambda x: np.sum(np.abs(x))
        elif args.normal_loss_ene == "MSELoss":
            self.loss_ene = lambda x: np.sum(x**2)
        else:
            raise ValueError(f"Unknown loss function {args.normal_loss_ene}")

        self.print = lambda msg: print(msg, flush=True) if self.verbose else None

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
        if args.md_number > 0:
            training_cycle_list.extend([f"_{i}" for i in range(1, args.md_number + 1)])

        for (
            name_mol,
            training_cycle_iteration,
        ) in product(
            molecule_list,
            training_cycle_list,
        ):
            name = f"{name_mol}_{args.basis}"

            try:
                mol = gen_mole(
                    name_mol,
                    args.basis,
                    dataset_name=args.dataset,
                    verbose=-1,
                )
                name = f"{name}{training_cycle_iteration}"

                path_name_ = DATA_PATH / f"data_{name}.npz"
                if not (path_name_).exists():
                    self.print(f"No file: {name:>40}")
                    error_molecule.append(name)
                    self.print(f"Error molecule: {error_molecule}")
                    continue

                name_list.append(name)
                if mol.natm == 1 and mol.charge == 0:
                    if atomic_name_dict is None:
                        self.atomic_name_dict[mol.atom_pure_symbol(0)] = name
                        self.print(f"{mol.elements} use {name}")

                mol_info_dict[name] = {
                    "natm": mol.natm,
                    "elements": mol.elements,
                    "charge": mol.charge,
                    "spin": mol.spin,
                    "nelec": mol.nelectron,
                }

            except ValueError as e:
                self.print(f"Error generating molecule {name}: {e}")

        # move atomic_name_dict in the head of name_list.
        for iter_atom_name, (atom_key, atom_name) in enumerate(
            self.atomic_name_dict.items()
        ):
            if atom_name in name_list:
                name_list.remove(atom_name)
                name_list.insert(iter_atom_name, atom_name)
            else:
                self.print(
                    f"Warning: atomic {atom_name} as {atom_key} is atom.",
                )
        self.print(name_list)

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
                    self.print(f"key : {key}, shape of val : {val.size()}")
                batch_gpu[key] = val[0].to(device=device, non_blocking=True)
                if PRINT_DEBUG:
                    self.print(
                        f"After processing, key : {key}, type of val : {type(batch_gpu[key])}, shape of val : {batch_gpu[key].size()}"
                    )
            else:
                if PRINT_DEBUG:
                    self.print(f"key : {key}, len of val : {len(val)}, val : {val}")
                if isinstance(val, list):
                    val_shape = np.shape(val)
                    if len(val_shape) == 1:
                        batch_gpu[key] = val[0]
                    elif len(val_shape) == 2:
                        batch_gpu[key] = list(np.array(val)[:, 0])
                    elif len(val_shape) == 3:
                        batch_gpu[key] = list(np.array(val)[:, :, 0])
                    elif len(val_shape) == 4:
                        batch_gpu[key] = list(np.array(val)[:, :, :, 0])
                    else:
                        raise ValueError(f"Unknown shape for key {key}: {val_shape}")
                else:
                    batch_gpu[key] = val[0]
                if PRINT_DEBUG:
                    self.print(
                        f"After processing, key : {key}, type of val : {type(batch_gpu[key])}, val : {batch_gpu[key]}",
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
        loss_multiplier = self.args.loss_multiplier
        loss_multiplier_abs = self.args.loss_multiplier_abs
        loss_multiplier_grad = self.args.loss_multiplier_grad
        loss_multiplier_atomic = self.args.loss_multiplier_atomic
        self.print(f"\nLoading data {name:<40}")
        data = np.load(DATA_PATH / f"data_{name}.npz", allow_pickle=True)

        weight_mat = data["weights"]
        if self.args.rho_input == "dft":
            input_mat = data["rho_cube_dft"]
            energy_target = data["e_cc"] - data["e_dft"]
        elif self.args.rho_input == "dft_d3bj":
            input_mat = data["rho_cube_dft"]
            energy_target = data["e_cc"] - data["e_dft_d3bj"]
        else:
            raise ValueError(f"Unknown rho_input: {self.args.rho_input}")

        data_dict = {
            "input": self.process_input(input_mat),
            "weight": weight_mat.reshape((-1, 1)),
        }

        if self.if_eval:
            data_dict["output"] = 0
            data_dict["grad2force"] = 0
            data_dict["grad_cc_train"] = 0
            if self.args.output_target == "b3lyp" and "exc_dft_grids" in data:
                output_mat = data["exc_dft_grids"]
                energy_target = np.sum(output_mat * weight_mat)
        else:
            if self.args.output_target == "tol_delta_grids":
                output_mat = data["tol_delta_grids"]
            elif self.args.output_target == "tol_delta_grids_l":
                output_mat = (
                    data["exc_cc_grids"]
                    + data["hatree_cc_grids"]
                    + data["kinl_cc_grids"]
                    + data["nuc_cc_grids"]
                ) - (
                    data["exc_dft_grids"]
                    + data["exc_k_dft_grids"]
                    + data["hatree_dft_grids"]
                    + data["kinl_dft_grids"]
                    + data["nuc_dft_grids"]
                )
            elif self.args.output_target == "tol_delta_grids_l_erf":
                output_mat = (
                    data["exc_cc_grids"]
                    + data["hatree_cc_grids"]
                    + data["kinl_cc_grids"]
                    + data["nuc_erf_cc_grids"]
                ) - (
                    data["exc_dft_grids"]
                    + data["exc_k_dft_grids"]
                    + data["hatree_dft_grids"]
                    + data["kinl_dft_grids"]
                    + data["nuc_erf_dft_grids"]
                )
            elif self.args.output_target == "exc_cc_grids":
                output_mat = (data["exc_cc_grids"]) - (
                    data["exc_dft_grids"] + data["exc_k_dft_grids"]
                )
            elif self.args.output_target == "b3lyp":
                output_mat = data["exc_dft_grids"]
                energy_target = np.sum(output_mat * weight_mat)
            else:
                raise ValueError(
                    f"Unknown output target: {self.args.output_target}",
                )

            grad2force = data["grad2force"]
            grad_cc_train = data["grad_cc_train"]

            if self.args.if_relative_weight_abs:
                loss_multiplier_abs /= self.loss_ene(np.abs(output_mat * weight_mat))
                self.print(
                    f"Adjusted loss_multiplier_abs: {loss_multiplier_abs:>6.3f}",
                )
                data_dict["loss_multiplier_abs"] = loss_multiplier_abs

            error_energy = AU2KCALMOL * abs(
                energy_target - np.sum(output_mat * weight_mat)
            )
            if np.abs(error_energy) > 0.01:
                print(
                    f"Warning: Large error energy {error_energy:>9.6f} kcal/mol "
                    f"for {name:>40} set to 0 in absolute loss calculation.",
                )
                data_dict["loss_multiplier_abs"] = 0

            data_dict["output"] = output_mat.reshape((-1, 1))
            data_dict["grad2force"] = self.process_grad2force(grad2force)
            data_dict["grad_cc_train"] = grad_cc_train

        atomic_systems = []
        atomic_stoichiometry = []
        num_data_used = mol_info["natm"]
        if num_data_used == 1:
            data_weight = self.args.atomic_weighting
        else:
            data_weight = np.sqrt(num_data_used)
        for i_atom in range(mol_info["natm"]):
            atom_name = mol_info["elements"][i_atom]
            if atom_name not in atomic_systems:
                atomic_systems.append(atom_name)
                atomic_stoichiometry.append(1)
            else:
                atomic_stoichiometry[atomic_systems.index(atom_name)] += 1

        ae_target = 0.0
        if self.args.if_atomic:
            if name in list(self.atomic_name_dict.values()):
                assert mol_info["natm"] == 1
                atom_name = mol_info["elements"][0]
                self.atomic_energy_dict[atom_name] = energy_target
            else:
                ae_target += energy_target
                for i_system in range(len(atomic_systems)):
                    system_atom = atomic_systems[i_system]
                    if system_atom in self.atomic_energy_dict:
                        ae_target -= (
                            atomic_stoichiometry[i_system]
                            * self.atomic_energy_dict[system_atom]
                        )
                    else:
                        self.print(
                            f"Warning: {system_atom} not found in atomic_name_dict, "
                            "skipping atomic energy calculation."
                        )
                        break

        data_dict["energy_target"] = energy_target
        data_dict["ae_target"] = ae_target
        data_dict["name"] = name
        data_dict["atomic_systems"] = atomic_systems
        data_dict["atomic_stoichiometry"] = atomic_stoichiometry
        data_dict["data_weight"] = data_weight

        if self.args.if_relative_weight:
            loss_multiplier /= self.loss_ene(energy_target)

            if np.abs(ae_target) < 1e-10:
                loss_multiplier_atomic = 0
            else:
                loss_multiplier_atomic /= self.loss_ene(ae_target)

        data_dict["loss_multiplier"] = loss_multiplier
        data_dict["loss_multiplier_abs"] = loss_multiplier_abs
        data_dict["loss_multiplier_grad"] = loss_multiplier_grad
        data_dict["loss_multiplier_atomic"] = loss_multiplier_atomic

        self.print(
            f"Adjusted loss_multiplier: {loss_multiplier:>6.3f}, loss_multiplier_grad {loss_multiplier_grad:>6.3f}, loss_multiplier_atomic {loss_multiplier_atomic:>6.3f}",
        )

        for key in self.array_key:
            data_dict[key] = torch.tensor(
                np.array(data_dict[key]), dtype=self.dtype, device="cpu"
            )

        return num_data_used, data_dict
