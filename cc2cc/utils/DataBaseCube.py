"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

import numpy as np
import torch

from cc2cc.utils.env_var import DATA_PATH, CUBE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.DataBase import DataBase


class DataBaseCube(DataBase):
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
        super().__init__(
            molecule_list,
            args,
            shuffle=shuffle,
            if_eval=if_eval,
            atomic_name_dict=atomic_name_dict,
            atomic_energy_dict=atomic_energy_dict,
        )

    def load_data(self, mol_info, name):
        """
        Load the data.
        """
        print("", flush=True)
        data = np.load(DATA_PATH / f"data_{name}.npz", allow_pickle=True)

        weight_mat = data["weights"]
        if self.args.rho_input == "dft":
            input_mat = data["rho_cube_dft"]
            output_mat = data["exc_cc_grids"]
            energy_target = data["energy_train"]
            grad2force = data["grad2force"]
            grad_cc_train = data["grad_cc_train"]
        elif self.args.rho_input == "cc":
            input_mat = data["rho_cube_cc"]
            output_mat = data["exc_cc_grids"]
            energy_target = data["energy_train"]
            grad2force = None
            grad_cc_train = None
        elif self.args.rho_input == "zmp":
            input_mat = data["rho_cube_zmp"]
            output_mat = data["exc_cc_grids_zmp"]
            energy_target = data["energy_train_zmp"]
            grad2force = None
            grad_cc_train = None
        else:
            raise ValueError(f"Unknown rho_input: {self.args.rho_input}")

        input_mat_index = (
            np.abs(input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]) > 1e-14
        )
        print(f"Total number of input points: {len(input_mat_index)}", flush=True)
        print(f"Number of non-zero input points: {np.sum(input_mat_index)}", flush=True)
        if len(output_mat.shape) != 0:
            print(
                f"Energy in zero input region: {AU2KCALMOL * np.sum(output_mat[~input_mat_index] * weight_mat[~input_mat_index])}",
                flush=True,
            )
            output_mat = output_mat[input_mat_index]
        if len(grad2force) != 0:
            grad2force = grad2force[:, :, input_mat_index, :]
        weight_mat = weight_mat[input_mat_index]
        input_mat = input_mat[input_mat_index]

        if not self.if_eval:
            error_energy = AU2KCALMOL * abs(
                energy_target - np.sum(output_mat * weight_mat)
            )
            print(f"Error energy {error_energy}: {name:>40}", flush=True)
            if error_energy > 1.5 * mol_info["natm"]:
                output_mat = torch.tensor(0)

        atomic_systems = []
        atomic_stoichiometry = []
        num_data_used = mol_info["natm"]
        for i_atom in range(mol_info["natm"]):
            atom_name = mol_info["elements"][i_atom]
            if atom_name not in atomic_systems:
                atomic_systems.append(atom_name)
                atomic_stoichiometry.append(1)
            else:
                atomic_stoichiometry[atomic_systems.index(atom_name)] += 1
        print(f"Total data used for {name}: {num_data_used}", flush=True)

        if self.args.if_atomic:
            ae_target = 0.0
            if name in self.atomic_name_dict:
                self.atomic_energy_dict[atom_name] = energy_target
            else:
                for i_system in range(len(atomic_systems)):
                    system_atom = atomic_systems[i_system]
                    if system_atom in self.atomic_energy_dict:
                        ae_target -= (
                            atomic_stoichiometry[i_system]
                            * self.atomic_energy_dict[system_atom]
                        )

            print(
                f"Atomic systems: {atomic_systems}, Stoichiometry: {atomic_stoichiometry}",
                flush=True,
            )

        if not self.if_eval and len(output_mat.shape) != 0:
            total_ene_used = np.sum(output_mat * weight_mat)
            total_ene_used_abs = np.sum(np.abs(output_mat * weight_mat))
            max_ene_den = np.max(output_mat * weight_mat)
            print(f"Total energy used: {AU2KCALMOL * total_ene_used}")
            print(f"Total abs energy used: {AU2KCALMOL * total_ene_used_abs}")
            print(f"Max energy density: {AU2KCALMOL * max_ene_den}")

        if self.args.if_relative_weight and not self.if_eval:
            loss_multiplier = self.args.loss_multiplier / self.loss_ene(
                torch.zeros_like(energy_target), energy_target
            )
            loss_multiplier_abs = self.args.loss_multiplier_abs / self.loss_ene_abs(
                torch.zeros_like(output_mat * weight_mat),
                torch.tensor(output_mat * weight_mat),
            )
            loss_multiplier_grad = self.args.loss_multiplier_grad / self.loss_grad(
                torch.zeros_like(grad_cc_train), grad_cc_train
            )
            loss_multiplier_atomic = (
                self.args.loss_multiplier_atomic
                / self.loss_ene_atomic(torch.zeros_like(ae_target), ae_target)
            )
        else:
            loss_multiplier = self.args.loss_multiplier
            loss_multiplier_abs = self.args.loss_multiplier_abs
            loss_multiplier_grad = self.args.loss_multiplier_grad
            loss_multiplier_atomic = self.args.loss_multiplier_atomic

        data_dict = {
            "input": torch.tensor(input_mat, dtype=self.dtype).detach().clone(),
            "weight": torch.tensor(weight_mat.reshape((-1, 1)), dtype=self.dtype)
            .detach()
            .clone(),
            "output": (
                torch.tensor(0).detach().clone()
                if self.if_eval
                else torch.tensor(output_mat.reshape((-1, 1)), dtype=self.dtype)
                .detach()
                .clone()
            ),
            "grad2force": (
                torch.tensor(0).detach().clone()
                if self.if_eval
                else torch.tensor(grad2force, dtype=self.dtype).detach().clone()
            ),
            "grad_cc_train": grad_cc_train,
            "energy_target": energy_target,
            "ae_target": ae_target,
            "name": name,
            "atomic_systems": atomic_systems,
            "atomic_stoichiometry": atomic_stoichiometry,
            "data_weight": (
                np.sqrt(40.0) if num_data_used == 1 else np.sqrt(num_data_used)
            ),
            "loss_multiplier": loss_multiplier,
            "loss_multiplier_abs": loss_multiplier_abs,
            "loss_multiplier_grad": loss_multiplier_grad,
            "loss_multiplier_atomic": loss_multiplier_atomic,
        }

        return num_data_used, data_dict
