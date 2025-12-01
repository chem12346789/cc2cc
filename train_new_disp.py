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

from cc2cc.utils import gen_mole
from cc2cc.utils import AU2KCALMOL


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, device="cuda", damping="zero", **kwargs):
        super().__init__()

        self.sizes = [self.d_model] + [self.mlp] * (self.depth - 1) + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size, bias=self.dense_bias)
                for input_size, output_size in zip(self.sizes, self.sizes[1:])
            ]
        )

        if self.dense_actv == "relu":
            self.actv_fn = nn.ReLU()
        elif self.dense_actv == "gelu":
            self.actv_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {self.actv_fn}")

        if self.dense_normal == "layer":
            self.norm = nn.ModuleList(
                [nn.LayerNorm(i_size) for i_size in self.sizes[:-2]]
            )
        elif self.dense_normal == "rms":
            self.norm = nn.ModuleList(
                [nn.RMSNorm(i_size) for i_size in self.sizes[:-2]]
            )
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in self.sizes[:-2]])

        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, batch_dicts):
        """
        Standard forward function, required for all nn.Module classes
        """
        for i_layer, layer in enumerate(self.layers):
            if self.if_skip_connection_dense:
                skip = x
            if i_layer < len(self.layers) - 1:
                x = self.norm[i_layer](x)
            x = layer(x)
            if i_layer < len(self.layers) - 1:
                x = self.actv_fn(x)
                x = self.dropout(x)
            if self.if_skip_connection_dense:
                if self.sizes[i_layer] == self.sizes[i_layer + 1]:
                    x = x + skip
        return x

    def obtain_batch_dicts(self, mol_list):

        batch_dicts = dict(
            Z=torch.cat([mol.charge for mol in mol_list], dim=0),  # (n_nodes,)
            pos=torch.cat(
                [mol.atom_coords() * units.Bohr for mol in mol_list], dim=0
            ),  # (n_nodes,)
            n_nodes_list=[mol.n_atoms for mol in mol_list],
        )

        batch_dicts["pos"].requires_grad_(True)
        return batch_dicts


if __name__ == "__main__":
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
    parser.add_argument(
        "--dataset", type=str, default="gmtkn-def2", help="Dataset name"
    )

    args = parser.parse_args()
    BASIS_SET = "def2-QZVP"
    DATA_PATH = f"validate_hkqai_done/ccdft_{BASIS_SET}_{args.load}_gmtkn-def2.csv"
    save_para = {}
    SAVE_PARA_PATH = (
        f"validate_hkqai_done/ccdft_{BASIS_SET}_{args.load}_gmtkn-def2.json"
    )

    # Set the random seed for reproducibility
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(
        "Warning: Using deterministic mode, which may slow down training.", flush=True
    )

    DEVICE = "cuda"

    data = pd.read_csv(DATA_PATH)

    with open("jupyter-notebook/subset.json", "r", encoding="utf-8") as f:
        # full_subset_dict = json.load(f)["full_subset_small_dict"]
        # full_subset_dict = json.load(f)["full_subset_dict"]
        full_subset_dict = json.load(f)["full_subset_dict_test"]
    name_mol_list = []
    for subset in full_subset_dict.values():
        for name_mol in subset:
            name_mol_list.append(f"{name_mol}")
    batch_subset = list(set(name_mol_list))
    print(batch_subset, flush=True)

    if Path(SAVE_PARA_PATH).exists():
        with open(SAVE_PARA_PATH, "r", encoding="utf-8") as f:
            load_para = json.load(f)
    else:
        load_para = None

    with open(
        "/home/chenzihao/workspace/cc2cc_test5/cc2cc/utils/gmtkn-def2.json",
        encoding="utf-8",
    ) as f:
        json_data = json.load(f)

    if args.load == "":
        dft_type_list = ["b3lyp"]
    else:
        dft_type_list = ["scf"]

    for damping, dft_type in product(["bj"], dft_type_list):
        data_name_list = (data["name"].str.split("_def2").str[0]).to_numpy()
        data_dft_ene = data[f"{dft_type}_ene"].to_numpy() * AU2KCALMOL

        input_batch = {}
        name_batch_list = {}
        weight_batch_list = {}
        mean_absolute_deviation = []

        if load_para is not None:
            load_para_disp = load_para[
                f"{"ai" if dft_type == "scf" else dft_type}_d3{damping}"
            ]
        else:
            load_para_disp = {}
        model = Model(device=DEVICE, damping=damping, **load_para_disp)
        model.compile(mode="max-autotune-no-cudagraphs")
        for name_mol in data_name_list:
            for i_subset in batch_subset:
                i_subset_name = "BH76" if i_subset == "BH76RC" else i_subset
                if name_mol.startswith(i_subset_name):
                    mol = gen_mole(
                        name_mol,
                        "def2-qzvp",
                        dataset_name=args.dataset,
                    )
                    input_batch[i_subset].append(mol)
                    if i_subset not in name_batch_list:
                        name_batch_list[i_subset] = []
                    name_batch_list[i_subset].append(name_mol)

        for i_subset in batch_subset:
            i_subset_name = "BH76" if i_subset == "BH76RC" else i_subset
            reaction_dict = json_data[f"reaction-{i_subset}"]
            name_batch_list[i_subset] = np.array(name_batch_list[i_subset])
            input_batch[i_subset] = model.obtain_batch_dicts(input_batch[i_subset])
            reaction_dict_copy = reaction_dict.copy()
            for i_reaction_name, (i_reaction_keys, i_reaction) in enumerate(
                reaction_dict_copy.items()
            ):
                systems_list = i_reaction["systems"]
                stoichiometry_list = i_reaction["stoichiometry"]

                for i in range(len(systems_list)):
                    mole_name = (
                        f"{systems_list[i]}"
                        if i_subset == "BH76RC"
                        else f"{i_subset}-{systems_list[i]}"
                    )
                    stoichiometry = int(stoichiometry_list[i])

                    if mole_name in json_data:
                        if isinstance(json_data[mole_name], str):
                            mole_name = json_data[mole_name]

                    col = np.where(data_name_list == mole_name)[0]
                    if col.size != 1:
                        print(
                            f"Warning: {mole_name} not found in name_list", flush=True
                        )
                        reaction_dict.pop(i_reaction_keys)
                        break
            json_data[f"reaction-{i_subset}"] = reaction_dict

        energy_batch_target = {}
        for i_subset in batch_subset:
            i_subset_name = "BH76" if i_subset == "BH76RC" else i_subset
            reaction_dict = json_data[f"reaction-{i_subset}"]

            energy_batch_target[i_subset] = torch.zeros(
                len(reaction_dict), dtype=torch.float64, device=DEVICE
            )
            weight_batch = np.zeros(len(reaction_dict), dtype=np.float64)
            for i_reaction_name, (i_reaction_keys, i_reaction) in enumerate(
                reaction_dict.items()
            ):
                systems_list = i_reaction["systems"]
                stoichiometry_list = i_reaction["stoichiometry"]
                weight_batch[i_reaction_name] = i_reaction["reference"]
                energy_batch_target[i_subset][i_reaction_name] = i_reaction["reference"]
                for i in range(len(systems_list)):
                    if i_subset == "BH76RC":
                        mole_name = f"{systems_list[i]}"
                    else:
                        mole_name = f"{i_subset}-{systems_list[i]}"
                    stoichiometry = int(stoichiometry_list[i])

                    if mole_name in json_data:
                        if isinstance(json_data[mole_name], str):
                            mole_name = json_data[mole_name]

                    col = np.where(data_name_list == mole_name)[0]
                    energy_batch_target[i_subset][i_reaction_name] += (
                        -data_dft_ene[col[0]]
                    ) * stoichiometry
            mean_absolute_deviation.extend(np.abs(weight_batch))
            weight_batch_list[i_subset] = 1 / np.mean(np.abs(weight_batch))

        print(
            f"mean_absolute_deviation: {56.84 / len(mean_absolute_deviation)}",
            flush=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-12,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,
            eta_min=1e-8,
        )
        loss_function = torch.nn.L1Loss(reduction="sum")
        torch.set_printoptions(precision=5, sci_mode=False)
        energy_batch_output = {}
        print("start training...", flush=True)

        parameter_list = []
        wtmad_2_list = []

        for epoch in tqdm.tqdm(range(args.epochs + 1)):
            loss_batch = []
            wtmad_2 = 0
            loss = 0.0
            optimizer.zero_grad()
            for i_subset in batch_subset:
                energy = model(input_batch[i_subset])

                reaction_dict = json_data[f"reaction-{i_subset}"]
                energy_batch_output[i_subset] = torch.zeros(
                    len(reaction_dict), dtype=torch.float64, device=DEVICE
                )
                for i_reaction_name, (i_reaction_keys, i_reaction) in enumerate(
                    reaction_dict.items()
                ):
                    systems_list = i_reaction["systems"]
                    stoichiometry_list = i_reaction["stoichiometry"]

                    for i in range(len(systems_list)):
                        mole_name = (
                            f"{systems_list[i]}"
                            if i_subset == "BH76RC"
                            else f"{i_subset}-{systems_list[i]}"
                        )
                        stoichiometry = int(stoichiometry_list[i])

                        if mole_name in json_data:
                            if isinstance(json_data[mole_name], str):
                                mole_name = json_data[mole_name]

                        col_disp = np.where(name_batch_list[i_subset] == mole_name)[0]
                        if col_disp.size == 1:
                            energy_batch_output[i_subset][i_reaction_name] += (
                                energy[col_disp[0]] * stoichiometry
                            )
                        else:
                            print(
                                f"Warning: {mole_name} not found in name_list",
                                flush=True,
                            )
                            break
                loss += (
                    loss_function(
                        energy_batch_output[i_subset], energy_batch_target[i_subset]
                    )
                    * weight_batch_list[i_subset]
                )
                loss_batch.append(
                    torch.mean(
                        torch.abs(
                            energy_batch_output[i_subset]
                            - energy_batch_target[i_subset]
                        )
                    ).item()
                )
                wtmad_2 += (
                    torch.sum(
                        torch.abs(
                            energy_batch_output[i_subset]
                            - energy_batch_target[i_subset]
                        )
                    )
                    * weight_batch_list[i_subset]
                ).item()

            # clip the loss to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                parameter_dict = {}
                for key, item in model.calc.dftd_module.params.items():
                    if isinstance(item, torch.Tensor):
                        parameter_dict[key] = item.detach().cpu().numpy().item()
                    else:
                        parameter_dict[key] = item
                parameter_list.append(deepcopy(parameter_dict))
                wtmad_2_list.append(wtmad_2 * 56.84 / len(mean_absolute_deviation))
                print(
                    f"Epoch: {epoch}, wtmad_2: {wtmad_2 * 56.84 / len(mean_absolute_deviation)}, loss: {loss_batch}",
                    flush=True,
                )

        best_epoch = np.argmin(wtmad_2_list)
        print(
            f"Best epoch: {best_epoch}, wtmad_2: {wtmad_2_list[best_epoch]}", flush=True
        )
        model_new = Model(device=DEVICE, damping=damping, **parameter_list[best_epoch])
        save_para[f"{"ai" if dft_type == "scf" else dft_type}_d3{damping}"] = (
            parameter_list[best_epoch]
        )
        data_dft_disp = []
        for name_mol in data_name_list:
            mol = gen_mole(
                name_mol,
                "def2-qzvp",
                dataset_name=args.dataset,
            )
            atoms = Atoms(
                symbols=mol.elements, positions=mol.atom_coords() * units.Bohr
            )
            energy = model_new(model_new.obtain_batch_dicts([atoms]))
            data_dft_disp.append(energy.item() / AU2KCALMOL)

        data[f"modified_{"ai" if dft_type == "scf" else dft_type}_d3{damping}"] = (
            data_dft_disp
        )

    data.to_csv(DATA_PATH, index=False)

    with open(SAVE_PARA_PATH, "w", encoding="utf-8") as f:
        json.dump(save_para, f)
