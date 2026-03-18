import json
from itertools import product
import argparse
from copy import deepcopy
from pathlib import Path
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

from ase import units
from ase.atoms import Atoms
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.calculators.dftd3 import DFTD3

from pyscf.data.elements import _std_symbol

from cc2cc.utils import AU2KCALMOL


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, device="cuda", damping="zero", **kwargs):
        super().__init__()

        # device="cuda:0" for fast GPU computation.
        self.calc = TorchDFTD3Calculator(
            device=device,
            dtype=torch.float64,
            xc="b3-lyp",
            damping=damping,
            bidirectional=False,
        )

        if damping == "zero":
            self.param_vector = torch.nn.Parameter(
                torch.tensor(
                    [
                        kwargs.get("rs6", 1.261),
                        kwargs.get("s18", 1.703),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
            )
            self.params = {
                "s6": kwargs.get("s6", 1.0),
                "rs6": self.param_vector[0],
                "s18": self.param_vector[1],
                "rs18": kwargs.get("rs18", 1.0),
                "alp": kwargs.get("alp", 14.0),
            }
        elif damping == "bj":
            self.param_vector = torch.nn.Parameter(
                torch.tensor(
                    # [
                    #     kwargs.get("s6", 1),
                    #     kwargs.get("rs6", 0.3981),
                    #     kwargs.get("s18", 1.9889),
                    #     kwargs.get("rs18", 4.4211),
                    # ],
                    [
                        kwargs.get("rs6", 10),
                        kwargs.get("rs18", 10),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
            )
            self.params = {
                "s6": 1,
                "rs6": self.param_vector[0],
                "s18": 1.9889,
                "rs18": self.param_vector[1],
                "alp": kwargs.get("alp", 14.0),
            }
        self.calc.dftd_module.params = self.params
        self.damping = damping

    def forward(self, batch_dicts):
        self.calc.reset()

        # Calculate the energy using the DFTD3 calculator
        E_disp = self.calc.dftd_module.calc_energy_batch(
            **batch_dicts, damping=self.damping
        )

        return E_disp * units.mol / units.kcal

    def obtain_batch_dicts(self, atoms_list):
        # Calculator.calculate(self, atoms, properties, system_changes)
        input_dicts_list = [self.calc._preprocess_atoms(atoms) for atoms in atoms_list]
        # --- Make batch ---
        n_nodes_list = [d["Z"].shape[0] for d in input_dicts_list]
        shift_index_array = torch.cumsum(torch.tensor([0] + n_nodes_list), dim=0)
        cell_batch = torch.stack(
            [
                (
                    torch.eye(3, device=self.calc.device, dtype=self.calc.dtype)
                    if d["cell"] is None
                    else d["cell"]
                )
                for d in input_dicts_list
            ]
        )

        batch_dicts = dict(
            Z=torch.cat([d["Z"] for d in input_dicts_list], dim=0),  # (n_nodes,)
            pos=torch.cat([d["pos"] for d in input_dicts_list], dim=0),  # (n_nodes,)
            cell=cell_batch,  # (bs, 3, 3)
            pbc=torch.stack([d["pbc"] for d in input_dicts_list]),  # (bs, 3)
            shift_pos=torch.cat(
                [d["shift_pos"] for d in input_dicts_list], dim=0
            ),  # (n_nodes,)
        )
        batch_dicts["edge_index"] = torch.cat(
            [
                d["edge_index"] + shift_index_array[i]
                for i, d in enumerate(input_dicts_list)
            ],
            dim=1,
        )
        batch_dicts["batch"] = torch.cat(
            [
                torch.full((n_nodes,), i, dtype=torch.long, device=self.calc.device)
                for i, n_nodes in enumerate(n_nodes_list)
            ],
            dim=0,
        )
        batch_dicts["batch_edge"] = torch.cat(
            [
                torch.full(
                    (d["edge_index"].shape[1],),
                    i,
                    dtype=torch.long,
                    device=self.calc.device,
                )
                for i, d in enumerate(input_dicts_list)
            ],
            dim=0,
        )

        batch_dicts["pos"].requires_grad_(True)
        return batch_dicts


parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
parser.add_argument(
    "--load", type=str, default="", help="Name of csv file, <csv_file_name>"
)
parser.add_argument(
    "--epochs", type=int, default=10000, help="Number of training epochs"
)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dataset", type=str, default="gmtkn-def2", help="Dataset name")
parser.add_argument("--basis", type=str, default="def2-QZVPP", help="Basis set")
parser.add_argument(
    "--damping", type=str, choices=["bj", "zero"], default="bj", help="Damping type"
)

args = parser.parse_args()
DATA_PATH = f"validate_hkqai_done/ccdft_{args.basis}_{args.load}_gmtkn-def2.csv"
save_para = {}
SAVE_PARA_PATH = f"validate_hkqai_done/ccdft_{args.basis}_{args.load}_gmtkn-def2.json"


# Set the random seed for reproducibility
random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cpu"

with open("jupyter-notebook/subset.json", "r", encoding="utf-8") as f:
    full_subset_dict = json.load(f)["full_subset_dict"]
batch_subset = []
for subset in full_subset_dict.values():
    for set_ in subset:
        batch_subset.append(set_)
print(batch_subset, flush=True)

model = Model(device=DEVICE, damping=args.damping)
model.compile(mode="max-autotune-no-cudagraphs")

data = pd.read_csv(DATA_PATH)
data_name_list = (data["name"].str.split("_def2").str[0]).to_numpy()
if args.load == "":
    dft_type = "b3lyp"
else:
    dft_type = "scf"
data_dft_ene = data[f"{dft_type}_ene"].to_numpy() * AU2KCALMOL


with open(
    "/home/chenzihao/workspace/cc2cc_test5/cc2cc/utils/gmtkn-def2.json",
    encoding="utf-8",
) as f:
    GMNTK55_json = json.load(f)
input_batch = []

for name_mol in data_name_list:
    molecule = GMNTK55_json[name_mol]
    symbols_list = [_std_symbol(symbol_coord[0]) for symbol_coord in molecule]
    coords_list = np.array([symbol_coord[1:] for symbol_coord in molecule])
    atoms = Atoms(symbols=symbols_list, positions=coords_list * units.Bohr)
    input_batch.append(atoms)

input_batch = model.obtain_batch_dicts(input_batch)


reference_energy = []
molecules_to_reactions = []
reactions_to_subset = []

for i_subset, subset_name in enumerate(batch_subset):
    i_subset_name = "BH76" if subset_name == "BH76RC" else subset_name
    reaction_dict = GMNTK55_json[f"reaction-{subset_name}"]
    for i_reaction_keys, i_reaction in reaction_dict.items():
        systems_list = i_reaction["systems"]
        stoichiometry_list = i_reaction["stoichiometry"]
        molecule_stoichiometry = torch.zeros(len(data_name_list))
        finished = True

        for i in range(len(systems_list)):
            mole_name = f"{i_subset_name}-{systems_list[i]}"
            stoichiometry = int(stoichiometry_list[i])

            if mole_name in GMNTK55_json:
                if isinstance(GMNTK55_json[mole_name], str):
                    mole_name = GMNTK55_json[mole_name]

            col = np.where(data_name_list == mole_name)[0]
            if col.size == 1:
                molecule_stoichiometry[col[0]] = stoichiometry
            else:
                finished = False

        if finished:
            reference_energy.append(i_reaction["reference"])
            molecules_to_reactions.append(molecule_stoichiometry)
            subset_index = torch.zeros(len(batch_subset))
            subset_index[i_subset] = 1
            reactions_to_subset.append(subset_index)

reference_energy = np.array(reference_energy)
molecules_to_reactions = np.array(molecules_to_reactions)
reactions_to_subset = np.array(reactions_to_subset)
number_of_reactions = np.sum(reactions_to_subset, axis=0)

average_relative_absolute_energies = np.einsum(
    "i,ij,j->j",
    np.abs(reference_energy),
    reactions_to_subset,
    1 / number_of_reactions,
)

reaction_energy = np.einsum("ji,i->j", molecules_to_reactions, data_dft_ene)
mean_absolute_deviation = 56.84 / 1505
mean_reaction_energy = np.einsum(
    "i,ij,j->j",
    np.abs(reference_energy - reaction_energy),
    reactions_to_subset,
    1 / number_of_reactions,
)
print(
    mean_absolute_deviation
    * np.einsum(
        "i,i,i->",
        number_of_reactions,
        1 / average_relative_absolute_energies,
        mean_reaction_energy,
    )
)
