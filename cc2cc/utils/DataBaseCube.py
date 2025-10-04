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

    def __init__(self, molecule_list, args, shuffle=True, if_eval=False):
        super().__init__(molecule_list, args, shuffle=shuffle, if_eval=if_eval)

    def load_data(self, mol_info, name):
        """
        Load the data.
        """
        print("", flush=True)
        data = np.load(DATA_PATH / f"data_{name}.npz", allow_pickle=True)

        weight_mat = data["weights"]
        if self.rho_input == "dft":
            input_mat = data["rho_cube_dft"]
            output_mat = data["exc_cc_grids"]
            energy_train = data["energy_train"]
            grad2force = data["grad2force"]
            grad_cc_train = data["grad_cc_train"]
        elif self.rho_input == "cc":
            input_mat = data["rho_cube_cc"]
            output_mat = data["exc_cc_grids"]
            energy_train = data["energy_train"]
            grad2force = None
            grad_cc_train = None
        elif self.rho_input == "zmp":
            input_mat = data["rho_cube_zmp"]
            output_mat = data["exc_cc_grids_zmp"]
            energy_train = data["energy_train_zmp"]
            grad2force = None
            grad_cc_train = None
        else:
            raise ValueError(f"Unknown rho_input: {self.rho_input}")

        input_mat_index = (
            np.abs(input_mat[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]) > 1e-10
        )
        print(f"Total number of input points: {len(input_mat_index)}", flush=True)
        print(f"Number of non-zero input points: {np.sum(input_mat_index)}", flush=True)
        if len(output_mat.shape) != 0:
            print(
                f"Energy in zero input region: {np.sum(output_mat[~input_mat_index] * weight_mat[~input_mat_index])}",
                flush=True,
            )
            output_mat = output_mat[input_mat_index]
        if len(grad2force) != 0:
            grad2force = grad2force[:, :, input_mat_index, :]
        weight_mat = weight_mat[input_mat_index]
        input_mat = input_mat[input_mat_index]

        if not self.if_eval:
            error_energy = AU2KCALMOL * abs(
                energy_train - np.sum(output_mat * weight_mat)
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

        data_dict = {
            "input": torch.tensor(input_mat, dtype=self.dtype),
            "weight": torch.tensor(weight_mat.reshape((-1, 1)), dtype=self.dtype),
            "output": (
                torch.tensor(0)
                if self.if_eval
                else torch.tensor(output_mat.reshape((-1, 1)), dtype=self.dtype)
            ),
            "grad2force": (
                torch.tensor(0)
                if self.if_eval
                else torch.tensor(grad2force, dtype=self.dtype)
            ),
            "grad_cc_train": grad_cc_train,
            "energy_train": energy_train,
            "name": name,
            "atomic_systems": atomic_systems,
            "atomic_stoichiometry": atomic_stoichiometry,
            "data_weight": np.sqrt(num_data_used) if num_data_used > 1 else 2.0,
        }

        return num_data_used, data_dict
