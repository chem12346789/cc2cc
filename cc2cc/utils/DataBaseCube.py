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
        verbose=False,
    ):
        super().__init__(
            molecule_list,
            args,
            shuffle=shuffle,
            if_eval=if_eval,
            atomic_name_dict=atomic_name_dict,
            atomic_energy_dict=atomic_energy_dict,
            verbose=verbose,
        )

    def load_data(self, mol_info, name):
        """
        Load the data.
        """
        self.print("")
        data = np.load(DATA_PATH / f"data_{name}.npz", allow_pickle=True)

        weight_mat = data["weights"]
        if self.args.rho_input == "dft":
            input_mat = data["rho_cube_dft"]
            output_mat = data["exc_cc_grids"]
            energy_target = data["energy_train"]
            grad2force = data["grad2force"]
            grad_cc_train = data["grad_cc_train"]
        else:
            raise ValueError(f"Unknown rho_input: {self.args.rho_input}")

        # input_mat_index = (
        #     np.abs(input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]) > 1e-10
        # )
        # self.print(f"Total number of input points: {len(input_mat_index)}")
        # self.print(f"Number of non-zero input points: {np.sum(input_mat_index)}")
        # if len(output_mat.shape) != 0:
        #     self.print(
        #         f"Energy in zero input region: {AU2KCALMOL * np.sum(output_mat[~input_mat_index] * weight_mat[~input_mat_index])}",
        #     )
        #     output_mat = output_mat[input_mat_index]
        # if len(grad2force) != 0:
        #     grad2force = grad2force[:, :, input_mat_index, :]
        # weight_mat = weight_mat[input_mat_index]
        # input_mat = input_mat[input_mat_index]

        self.print("")
        self.print("After filtering:")
        self.print(f"max input value: {np.max(input_mat)} at {np.argmax(input_mat)}")
        self.print(f"min input value: {np.min(input_mat)} at {np.argmin(input_mat)}")
        if not self.if_eval and len(output_mat.shape) != 0:
            self.print(
                f"max output value: {np.max(output_mat)} at {np.argmax(output_mat)}"
            )
            self.print(
                f"min output value: {np.min(output_mat)} at {np.argmin(output_mat)}"
            )
        self.print(
            f"max input value with weight: {np.max(np.einsum('pcxyz,p->pcxyz', input_mat, weight_mat))}"
        )
        self.print(
            f"min input value with weight: {np.min(np.einsum('pcxyz,p->pcxyz', input_mat, weight_mat))}"
        )
        if not self.if_eval and len(output_mat.shape) != 0:
            self.print(
                f"max output value with weight: {np.max(output_mat * weight_mat)}"
            )
            self.print(
                f"min output value with weight: {np.min(output_mat * weight_mat)}"
            )

        b3lyp_ene = (
            0.08 * input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.19 * input_mat[:, 1, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.72 * input_mat[:, 2, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.81 * input_mat[:, 3, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        )
        self.print(
            f"Input shape after filtering: {input_mat.shape};"
            f"B3lyp_ene shape after filtering: {b3lyp_ene.shape};"
            f"Weight shape after filtering: {weight_mat.shape}",
        )
        mean_val = np.sum(b3lyp_ene * weight_mat)
        normal_factor = np.sqrt(
            np.sqrt(np.sum((b3lyp_ene - mean_val) ** 2 * weight_mat) / mol_info["natm"])
        )
        self.print(f"Normal factor: {normal_factor}")

        if not self.if_eval:
            error_energy = AU2KCALMOL * abs(
                energy_target - np.sum(output_mat * weight_mat)
            )
            self.print(f"Error energy {error_energy}: {name:>40}")

        atomic_systems = []
        atomic_stoichiometry = []
        num_data_used = mol_info["natm"]
        data_weight = self.args.atomic_weighting
        if num_data_used == 1:
            data_weight = 1
        else:
            data_weight = np.sqrt(num_data_used)
        for i_atom in range(mol_info["natm"]):
            atom_name = mol_info["elements"][i_atom]
            if atom_name not in atomic_systems:
                atomic_systems.append(atom_name)
                atomic_stoichiometry.append(1)
            else:
                atomic_stoichiometry[atomic_systems.index(atom_name)] += 1
        self.print(f"Total data used for {name}: {num_data_used}")

        ae_target = 0.0
        if self.args.if_atomic:
            if name in list(self.atomic_name_dict.values()):
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

            self.print(
                f"Atomic systems: {atomic_systems}, Stoichiometry: {atomic_stoichiometry} , AE target: {ae_target * AU2KCALMOL}",
            )

        if not self.if_eval and len(output_mat.shape) != 0:
            total_ene_used = np.sum(output_mat * weight_mat)
            total_ene_used_abs = np.sum(np.abs(output_mat * weight_mat))
            max_ene_den = np.max(output_mat * weight_mat)
            self.print(f"Total energy used: {AU2KCALMOL * total_ene_used}")
            self.print(f"Total abs energy used: {AU2KCALMOL * total_ene_used_abs}")
            self.print(f"Max energy density: {AU2KCALMOL * max_ene_den}")

        loss_multiplier = self.args.loss_multiplier
        loss_multiplier_abs = self.args.loss_multiplier_abs
        loss_multiplier_grad = self.args.loss_multiplier_grad
        loss_multiplier_atomic = self.args.loss_multiplier_atomic

        if self.args.if_relative_weight and not self.if_eval:
            epsilon = 1e-10
            loss_multiplier = self.args.loss_multiplier / (
                self.loss_ene(
                    torch.zeros(()),
                    AU2KCALMOL * torch.tensor(energy_target),
                )
                + epsilon
            )
            # loss_multiplier_abs = self.args.loss_multiplier_abs / (
            #     self.loss_ene_abs(
            #         torch.zeros((output_mat * weight_mat).shape),
            #         AU2KCALMOL * torch.tensor(output_mat * weight_mat),
            #     )
            #     + epsilon
            # )
            if grad_cc_train is not None:
                loss_multiplier_grad = self.args.loss_multiplier_grad / (
                    self.loss_grad(
                        torch.zeros(grad_cc_train.shape),
                        AU2KCALMOL * torch.tensor(grad_cc_train),
                    )
                    + epsilon
                )
            else:
                loss_multiplier_grad = 1
            loss_multiplier_atomic = self.args.loss_multiplier_atomic / (
                self.loss_ene_atomic(
                    torch.zeros(()), AU2KCALMOL * torch.tensor(ae_target)
                )
            )
            if loss_multiplier_atomic > 1 / epsilon:
                loss_multiplier_atomic = 0
            self.print(
                f"Relative loss multipliers: {loss_multiplier}, {loss_multiplier_abs}, {loss_multiplier_grad}, {loss_multiplier_atomic}",
            )

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
            "normal_factor": normal_factor,
            "data_weight": data_weight,
            "loss_multiplier": loss_multiplier,
            "loss_multiplier_abs": loss_multiplier_abs,
            "loss_multiplier_grad": loss_multiplier_grad,
            "loss_multiplier_atomic": loss_multiplier_atomic,
        }

        self.print("")
        return num_data_used, data_dict
