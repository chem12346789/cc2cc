"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

from itertools import product
from collections import Counter
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cc2cc.utils.mol import gen_mole
from cc2cc.utils.env_var import DATA_PATH
from cc2cc.utils.mol import AU2KCALMOL

EPS = 1e-2
MAX_ERROR_ENERGY = 0.01  # kcal/mol per atom, if the error energy is larger than this value, we set the absolute loss multiplier to 0 to avoid the numerical instability in training.
MAX_GRAD_BATCH_SIZE = 169000  # if the number of gradients is larger than this value, we split the batch into smaller batches to avoid the memory overflow in training.


class BasicDataset(Dataset):
    """
    Documentation for a class.
    """

    def __init__(self, name_list, mol_info_dict, load_data):
        super().__init__()
        self.data = {}
        self.name_list = []
        total_number_of_atom = 0

        for name in name_list:
            num_data_used, data_dict = load_data(mol_info_dict[name], name)
            total_number_of_atom += num_data_used
            if num_data_used != 0:
                self.data[name] = data_dict
                self.name_list.append(name)

            # Add more copies of the atomic data to balance the dataset.
            # This is useful when we need to have more data for single-atom systems.
            if num_data_used == 1:
                append_number = max(40 // int(data_dict["data_weight"]) - 1, 0)
                self.name_list.extend([name] * append_number)
                total_number_of_atom += num_data_used * append_number

        print(
            f"Total number of data: {len(self.name_list)}, total number of atoms: {total_number_of_atom}"
        )

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
        self.gpu_key = (
            "input",
            "weight",
            "output",
            "energy_target",
            "ae_target",
            "grad2force",
            "grad_cc_train",
        )

        loss_func_dict = {
            "L1Loss": lambda _: 1,
            "MSELoss": lambda x: np.sum(np.abs(x)),
        }
        loss_func_inversed_dict = {
            "L1Loss": lambda _: 1,
            "MSELoss": lambda x: np.sum(np.sqrt(x)),
        }
        self.loss_ene = loss_func_dict[args.loss_type]
        self.loss_ene_inversed = loss_func_inversed_dict[args.loss_type]

        name_list = []
        error_molecule = []
        mol_info_dict = {}
        self.atomic_name_dict = {} if atomic_name_dict is None else atomic_name_dict
        self.atomic_energy_dict = (
            {} if atomic_energy_dict is None else atomic_energy_dict
        )

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
        self.atomic_name_values = set(self.atomic_name_dict.values())
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
            batch_size=None,
            num_workers=0,
            pin_memory=True,
            sampler=self.sampler,
        )

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __len__(self):
        return len(self.dataset.name_list)

    def process_batch(self, batch, device: str | int = "cuda"):
        """
        Load the batch data to the GPU.
        Note all data is in the list ([data]), so we need to access the first element.
        """

        batch_gpu = {}
        for key, val in batch.items():
            if key in self.gpu_key:
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
        elif self.args.rho_input == "zmp":
            if self.if_eval:
                input_mat = data["rho_cube_dft"]
                energy_target = data["e_cc"] - data["e_dft"]
            else:
                input_mat = data["rho_cube_zmp"]
                energy_target = data["e_cc"] - data["e_zmp"]
        else:
            raise ValueError(f"Unknown rho_input: {self.args.rho_input}")

        data_dict = {
            "input": self.process_input(input_mat),
            "weight": weight_mat.reshape((-1, 1)),
        }

        # if the input_mat is too large, we filter the columns with small values to avoid the numrical instability in training. We keep the columns with the sum of absolute values larger than 1e-15.
        # if you want to keep all the data, you can set the threshold to 0.
        # if you want to save more memory, you can set the threshold to a larger value.
        if self.if_eval:
            filter_idx = np.sum(np.abs(data_dict["input"]), axis=(1, 2)) > 1e-10
        else:
            filter_idx = np.sum(np.abs(data_dict["input"]), axis=(1, 2)) > 1e-15
        data_dict["input"] = data_dict["input"][filter_idx]
        data_dict["weight"] = data_dict["weight"][filter_idx]
        del input_mat, weight_mat

        if self.if_eval:
            data_dict["output"] = 0
            data_dict["grad2force"] = 0
            data_dict["grad_cc_train"] = 0
        else:
            if self.args.output_target == "tol_delta_grids":
                if self.args.rho_input in ("dft", "dft_d3bj"):
                    output_mat = data["tol_delta_grids"]
                elif self.args.rho_input == "zmp":
                    output_mat = data["tol_delta_zmp_grids"]
                else:
                    raise ValueError(f"Unknown rho_input: {self.args.rho_input}")
            else:
                raise ValueError(f"Unknown output_target: {self.args.output_target}")
            data_dict["output"] = output_mat.reshape((-1, 1))[filter_idx]
            del output_mat

            if self.args.if_abs:
                if self.args.if_relative_weight_abs:
                    loss_multiplier_abs /= (
                        self.loss_ene(data_dict["output"] * data_dict["weight"]) + EPS
                    )
                error_energy = AU2KCALMOL * abs(
                    energy_target - np.sum(data_dict["output"] * data_dict["weight"])
                )
                if error_energy > MAX_ERROR_ENERGY * mol_info["natm"]:
                    self.print(
                        f"Warning: Large error energy {error_energy:>9.6f} kcal/mol "
                        f"for {name:>40} set to 0 in absolute loss calculation.",
                    )
                    loss_multiplier_abs = 0

            if self.args.if_grad and len(data_dict["input"]) < MAX_GRAD_BATCH_SIZE:
                grad2force = data["grad2force"]
                grad_cc_train = data["grad_cc_train"]
                data_dict["grad2force"] = self.process_grad2force(grad2force)[
                    filter_idx
                ]
                data_dict["grad_cc_train"] = grad_cc_train.reshape(-1)
                loss_multiplier_grad /= self.loss_ene(grad_cc_train) + EPS
            else:
                data_dict["grad2force"] = 0
                data_dict["grad_cc_train"] = 0

        element_counts = Counter(mol_info["elements"])
        atomic_systems = list(element_counts.keys())
        atomic_stoichiometry = list(element_counts.values())
        num_data_used = mol_info["natm"]
        if num_data_used == 1:
            data_weight = self.args.atomic_weighting
        else:
            data_weight = num_data_used
        self.print(f"data_weight: {data_weight:>6.3f}")

        ae_target = 0.0
        if self.args.if_atomic:
            if name in self.atomic_name_values:
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
            loss_multiplier /= self.loss_ene(energy_target) + EPS
            loss_multiplier_atomic /= self.loss_ene(ae_target) + EPS
        if abs(ae_target) < 1e-10:
            self.print(
                f"Warning: ae_target is too small {ae_target:>6.3f} for {name:>40}, set to 0.0 to turn it off in the atomic loss calculation.",
            )
            loss_multiplier_atomic = 0

        data_dict["loss_multiplier"] = data_weight * loss_multiplier
        data_dict["loss_multiplier_abs"] = data_weight * loss_multiplier_abs
        data_dict["loss_multiplier_grad"] = data_weight * loss_multiplier_grad
        data_dict["loss_multiplier_atomic"] = data_weight * loss_multiplier_atomic

        self.print(
            f"Adjusted loss_multiplier: {loss_multiplier:>6.3f}, grad {loss_multiplier_grad:>6.3f}, atomic {loss_multiplier_atomic:>6.3f}, abs {loss_multiplier_abs:>6.3f}",
        )

        for key in self.gpu_key:
            if isinstance(data_dict[key], np.ndarray):
                self.print(f"key: {key}, shape: {data_dict[key].shape}")
            data_dict[key] = torch.as_tensor(data_dict[key], dtype=self.dtype)

        return num_data_used, data_dict
