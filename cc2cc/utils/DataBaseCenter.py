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

    def __init__(self, molecule_list, args, shuffle=True):
        super().__init__(molecule_list, args, shuffle=shuffle)

    def load_data(self, mol_info, name):
        """
        Load the data.
        """
        print("", flush=True)
        data = np.load(DATA_PATH / f"data_{name}.npz")

        weight_mat = data["weights"]
        if self.rho_input == "dft":
            input_mat = data["rho_cube_dft"]
            output_mat = data["exc_cc_grids"]
            error_energy = data["error_energy"]
        elif self.rho_input == "cc":
            input_mat = data["rho_cube_cc"]
            output_mat = data["exc_cc_grids"]
            error_energy = data["error_energy"]
        elif self.rho_input == "zmp":
            input_mat = data["rho_cube_zmp"]
            output_mat = data["exc_cc_grids_zmp"]
            error_energy = data["error_energy_zmp"]
        else:
            raise ValueError(f"Unknown rho_input: {self.rho_input}")

        # print(f"Total energy real: {AU2KCALMOL * data['error_energy']}")
        # print(f"Total energy: {AU2KCALMOL * np.sum(output_mat * weight_mat)}")
        if (
            AU2KCALMOL * abs(error_energy - np.sum(output_mat * weight_mat))
            > 0.2 * mol_info["natm"]
        ):
            print(f"Error energy is too large: {name:>40}", flush=True)
            return 0, {}

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
            input_.append(input_mat[slice_, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE])
            weight_.append(weight_mat[slice_])
            output_.append(output_mat[slice_])
            total_ene_used += np.sum(output_mat[slice_] * weight_mat[slice_])
        input_ = np.array(input_).reshape((-1, 4))
        weight_ = np.array(weight_).reshape((-1, 1))
        output_ = np.array(output_).reshape((-1, 1))

        if num_data_used == 0:
            return 0, {}

        print(f"Total energy used: {AU2KCALMOL * total_ene_used}")
        print(f"Total data used for {name}: {num_data_used}", flush=True)
        print(
            f"Atomic systems: {atomic_systems}, Stoichiometry: {atomic_stoichiometry}",
            flush=True,
        )

        data_dict = {
            "input": torch.tensor(input_, dtype=self.dtype),
            "weight": torch.tensor(weight_, dtype=self.dtype),
            "output": torch.tensor(output_, dtype=self.dtype),
            "error_energy": error_energy,
            "name": name,
            "atomic_systems": atomic_systems,
            "atomic_stoichiometry": atomic_stoichiometry,
            "data_weight": np.sqrt(num_data_used) if num_data_used > 0 else 0,
        }

        return num_data_used, data_dict
