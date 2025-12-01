"""
Module for handling molecular data and generating datasets for machine learning tasks, for center data.
"""

import numpy as np
import torch

from cc2cc.utils.env_var import DATA_PATH, CUBE_MIDDLE
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.DataBase import DataBase


class DataBaseCenter(DataBase):
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
        )

    def load_data(self, mol_info, name):
        """
        Load the data.
        """
        print("", flush=True)
        data = np.load(DATA_PATH / f"data_{name}.npz")

        weight_mat = data["weights"]
        if self.args.rho_input == "dft":
            input_mat = data["rho_cube_dft"]
            output_mat = data["exc_cc_grids"]
            energy_train = data["energy_train"]
            grad2force = data["grad2force"]
            grad_cc_train = data["grad_cc_train"]
        else:
            raise ValueError(f"Unknown rho_input: {self.args.rho_input}")

        if not self.if_eval:
            error_energy = AU2KCALMOL * abs(
                energy_train - np.sum(output_mat * weight_mat)
            )
            if error_energy > 0.2 * mol_info["natm"]:
                print(
                    f"Error energy {error_energy} is too large: {name:>40}", flush=True
                )

        num_data_used = mol_info["natm"]
        total_ene_used = np.sum(output_mat * weight_mat)
        total_ene_used_abs = np.sum(np.abs(output_mat * weight_mat))
        max_ene_den = np.max(output_mat * weight_mat)

        atomic_systems = []
        atomic_stoichiometry = []
        for i_atom in range(mol_info["natm"]):
            atom_name = mol_info["elements"][i_atom]
            if atom_name not in atomic_systems:
                atomic_systems.append(atom_name)
                atomic_stoichiometry.append(1)
            else:
                atomic_stoichiometry[atomic_systems.index(atom_name)] += 1

        print(f"Total energy used: {AU2KCALMOL * total_ene_used}")
        print(f"Total abs energy used: {AU2KCALMOL * total_ene_used_abs}")
        print(f"Max energy density: {AU2KCALMOL * max_ene_den}")
        print(f"Total data used for {name}: {num_data_used}", flush=True)
        print(
            f"Atomic systems: {atomic_systems}, Stoichiometry: {atomic_stoichiometry}",
            flush=True,
        )

        data_dict = {
            "input": torch.tensor(
                input_mat[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE], dtype=self.dtype
            ),
            "weight": torch.tensor(weight_mat, dtype=self.dtype),
            "output": (
                torch.tensor(0)
                if self.if_eval
                else torch.tensor(output_mat, dtype=self.dtype)
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
            "data_weight": np.sqrt(num_data_used) if num_data_used > 0 else 0,
        }

        return num_data_used, data_dict
