"""
Module for handling molecular data and generating datasets for machine learning tasks, for cube data.
"""

import numpy as np
import torch

from cc2cc.utils.env_var import DATA_PATH, CUBE_SIZE
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
        elif self.rho_input == "cc":
            input_mat = data["rho_cube_cc"]
            output_mat = data["exc_cc_grids"]
            energy_train = data["energy_train"]
        elif self.rho_input == "zmp":
            input_mat = data["rho_cube_zmp"]
            output_mat = data["exc_cc_grids_zmp"]
            energy_train = data["energy_train_zmp"]
        else:
            raise ValueError(f"Unknown rho_input: {self.rho_input}")

        if not self.if_eval:
            error_energy = AU2KCALMOL * abs(
                energy_train - np.sum(output_mat * weight_mat)
            )
            if error_energy > 0.2 * mol_info["natm"]:
                print(
                    f"Error energy {error_energy} is too large: {name:>40}", flush=True
                )

        input_ = []
        weight_ = []
        output_ = []
        atomic_systems = []
        atomic_stoichiometry = []

        num_data_used = 0
        total_ene_used = 0
        total_ene_used_abs = 0
        max_ene_den = 0
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
            if not self.if_eval:
                output_.append(output_mat[slice_])
                total_ene_used += np.sum(output_mat[slice_] * weight_mat[slice_])
                total_ene_used_abs += np.sum(
                    np.abs(output_mat[slice_] * weight_mat[slice_])
                )
                max_ene_den = max(
                    max_ene_den, np.max(output_mat[slice_] * weight_mat[slice_])
                )
        input_ = np.array(input_).reshape((-1, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        weight_ = np.array(weight_).reshape((-1, 1))
        if not self.if_eval:
            output_ = np.array(output_).reshape((-1, 1))

        if num_data_used == 0:
            return 0, {}

        print(f"Total energy used: {AU2KCALMOL * total_ene_used}")
        print(f"Total abs energy used: {AU2KCALMOL * total_ene_used_abs}")
        print(f"Max energy density: {AU2KCALMOL * max_ene_den}")
        print(f"Total data used for {name}: {num_data_used}", flush=True)
        print(
            f"Atomic systems: {atomic_systems}, Stoichiometry: {atomic_stoichiometry}",
            flush=True,
        )

        data_dict = {
            "input": torch.tensor(input_, dtype=self.dtype),
            "weight": torch.tensor(weight_, dtype=self.dtype),
            "output": (
                torch.tensor(0)
                if self.if_eval
                else torch.tensor(output_, dtype=self.dtype)
            ),
            "energy_train": energy_train,
            "name": name,
            "atomic_systems": atomic_systems,
            "atomic_stoichiometry": atomic_stoichiometry,
            "data_weight": np.sqrt(num_data_used) if num_data_used > 0 else 0,
        }

        return num_data_used, data_dict
