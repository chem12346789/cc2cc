import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ase import units
from ase.atoms import Atoms
from pyscf.data.elements import _std_symbol
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from cc2cc.utils import AU2KCALMOL


DTYPE = torch.float64
DEFAULT_DEVICE = "cuda"
SUBSET_JSON_PATH = Path("jupyter-notebook/subset.json")
DATASET_JSON_DIR = Path("cc2cc/utils")
VALIDATE_DIR = Path("validate_hkqai_done")


@dataclass
class ReactionTensors:
    reference_energy: torch.Tensor
    molecules_to_reactions: torch.Tensor
    reactions_to_subset: torch.Tensor
    number_of_reactions: torch.Tensor
    average_relative_absolute_energies: torch.Tensor


class D3Model(nn.Module):
    def __init__(self, device: str, damping: str, initial_params: Optional[Dict[str, float]] = None):
        super().__init__()
        initial_params = initial_params or {}

        self.calc = TorchDFTD3Calculator(
            device=device,
            dtype=DTYPE,
            xc="b3-lyp",
            damping=damping,
            bidirectional=False,
        )
        self.damping = damping

        if damping == "zero":
            self.param_vector = nn.Parameter(
                torch.tensor(
                    [
                        initial_params.get("s6", 0.0),
                        initial_params.get("rs6", 1.261),
                        initial_params.get("s18", 0.0),
                    ],
                    dtype=DTYPE,
                    device=device,
                )
            )
            self.params = {
                "s6": self.param_vector[0],
                "rs6": self.param_vector[1],
                "s18": self.param_vector[2],
                "rs18": initial_params.get("rs18", 1.0),
                "alp": initial_params.get("alp", 14.0),
            }
        elif damping == "bj":
            self.param_vector = nn.Parameter(
                torch.tensor(
                    [
                        initial_params.get("rs6", 1),
                        initial_params.get("s18", 2),
                        initial_params.get("rs18", 4),
                    ],
                    dtype=DTYPE,
                    device=device,
                )
            )
            self.params = {
                "s6": 1.0,
                "rs6": self.param_vector[0],
                "s18": self.param_vector[1],
                "rs18": self.param_vector[2],
                "alp": 14.0,
            }
        else:
            raise ValueError(f"Unsupported damping: {damping}")

        self.calc.dftd_module.params = self.params

    def current_params(self) -> Dict[str, float]:
        out = {}
        for k, v in self.params.items():
            out[k] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
        return out

    def dispersion_energy_kcalmol(self, batch_dicts: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.calc.reset()
        e_disp = self.calc.dftd_module.calc_energy_batch(**batch_dicts, damping=self.damping)
        return e_disp * units.mol / units.kcal

    def total_energy_hartree(self, batch_dicts: Dict[str, torch.Tensor], scf_ene_au: np.ndarray) -> np.ndarray:
        self.calc.reset()
        e_disp = self.calc.dftd_module.calc_energy_batch(**batch_dicts, damping=self.damping)
        return e_disp.detach().cpu().numpy() / units.Hartree + scf_ene_au

    def obtain_batch_dicts(self, atoms_list: Iterable[Atoms]) -> Dict[str, torch.Tensor]:
        input_dicts_list = [self.calc._preprocess_atoms(atoms) for atoms in atoms_list]
        n_nodes_list = [d["Z"].shape[0] for d in input_dicts_list]
        shift_index_array = torch.cumsum(torch.tensor([0] + n_nodes_list), dim=0)

        cell_batch = torch.stack(
            [
                torch.eye(3, device=self.calc.device, dtype=self.calc.dtype)
                if d["cell"] is None
                else d["cell"]
                for d in input_dicts_list
            ]
        )

        batch_dicts = {
            "Z": torch.cat([d["Z"] for d in input_dicts_list], dim=0),
            "pos": torch.cat([d["pos"] for d in input_dicts_list], dim=0),
            "cell": cell_batch,
            "pbc": torch.stack([d["pbc"] for d in input_dicts_list]),
            "shift_pos": torch.cat([d["shift_pos"] for d in input_dicts_list], dim=0),
            "edge_index": torch.cat(
                [d["edge_index"] + shift_index_array[i] for i, d in enumerate(input_dicts_list)],
                dim=1,
            ),
            "batch": torch.cat(
                [
                    torch.full((n_nodes,), i, dtype=torch.long, device=self.calc.device)
                    for i, n_nodes in enumerate(n_nodes_list)
                ],
                dim=0,
            ),
            "batch_edge": torch.cat(
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
            ),
        }
        batch_dicts["pos"].requires_grad_(True)
        return batch_dicts


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_subset(subset_dict: Dict[str, List[str]]) -> List[str]:
    names = []
    for subsets in subset_dict.values():
        names.extend(subsets)
    return names


def subset_alias(name: str) -> str:
    if name == "BH76RC":
        return "BH76"
    if "S66x8" in name:
        return "S66x8"
    if "S22x5" in name:
        return "S22x5"
    return name


def build_atoms(data_names: np.ndarray, dataset_json: Dict) -> List[Atoms]:
    atoms_list = []
    for name in data_names:
        molecule = dataset_json[name]
        symbols = [_std_symbol(symbol_coord[0]) for symbol_coord in molecule]
        coords = np.array([symbol_coord[1:] for symbol_coord in molecule])
        atoms_list.append(Atoms(symbols=symbols, positions=coords))
    return atoms_list


def build_reaction_tensors(
    batch_subset: List[str],
    dataset_json: Dict,
    data_name_list: np.ndarray,
    device: str,
) -> ReactionTensors:
    reference_energy = []
    molecules_to_reactions = []
    reactions_to_subset = []

    for i_subset, subset_name in enumerate(batch_subset):
        reaction_dict = dataset_json[f"reaction-{subset_name}"]
        prefix = subset_alias(subset_name)

        for _, reaction in reaction_dict.items():
            systems = reaction["systems"]
            stoich = reaction["stoichiometry"]
            molecule_stoichiometry = torch.zeros(len(data_name_list))
            finished = True

            for i in range(len(systems)):
                molecule_name = f"{prefix}-{systems[i]}"
                coeff = int(stoich[i])

                if molecule_name in dataset_json and isinstance(dataset_json[molecule_name], str):
                    molecule_name = dataset_json[molecule_name]

                col = np.where(data_name_list == molecule_name)[0]
                if col.size == 1:
                    molecule_stoichiometry[col[0]] = coeff
                else:
                    finished = False

            if finished:
                reference_energy.append(reaction["reference"])
                molecules_to_reactions.append(molecule_stoichiometry)
                subset_index = torch.zeros(len(batch_subset))
                subset_index[i_subset] = 1
                reactions_to_subset.append(subset_index)

    reference_energy_t = torch.tensor(np.array(reference_energy), device=device, dtype=DTYPE)
    molecules_to_reactions_t = torch.tensor(np.array(molecules_to_reactions), device=device, dtype=DTYPE)
    reactions_to_subset_t = torch.tensor(np.array(reactions_to_subset), device=device, dtype=DTYPE)

    number_of_reactions = torch.sum(reactions_to_subset_t, axis=0)
    average_relative_absolute_energies = torch.einsum(
        "i,ij,j->j",
        torch.abs(reference_energy_t),
        reactions_to_subset_t,
        1 / number_of_reactions,
    )
    return ReactionTensors(
        reference_energy=reference_energy_t,
        molecules_to_reactions=molecules_to_reactions_t,
        reactions_to_subset=reactions_to_subset_t,
        number_of_reactions=number_of_reactions,
        average_relative_absolute_energies=average_relative_absolute_energies,
    )


def train_loss(
    model: D3Model,
    input_batch: Dict[str, torch.Tensor],
    reaction_tensors: ReactionTensors,
    data_dft_ene_kcalmol: torch.Tensor,
) -> torch.Tensor:
    e_disp_kcalmol = model.dispersion_energy_kcalmol(input_batch)
    reaction_energy = torch.einsum(
        "ji,i->j",
        reaction_tensors.molecules_to_reactions,
        e_disp_kcalmol + data_dft_ene_kcalmol,
    )
    return torch.mean(torch.abs(reaction_tensors.reference_energy - reaction_energy))


def eval_loss_weighted(
    model: D3Model,
    input_batch: Dict[str, torch.Tensor],
    reaction_tensors: ReactionTensors,
    data_dft_ene_kcalmol: torch.Tensor,
) -> torch.Tensor:
    mean_absolute_deviation = 56.84 / 1505
    e_disp_kcalmol = model.dispersion_energy_kcalmol(input_batch)

    reaction_energy = torch.einsum(
        "ji,i->j",
        reaction_tensors.molecules_to_reactions,
        e_disp_kcalmol + data_dft_ene_kcalmol,
    )
    mean_reaction_energy = torch.einsum(
        "i,ij,j->j",
        torch.abs(reaction_tensors.reference_energy - reaction_energy),
        reaction_tensors.reactions_to_subset,
        1 / reaction_tensors.number_of_reactions,
    )
    return torch.abs(
        mean_absolute_deviation
        * torch.einsum(
            "i,i,i->",
            reaction_tensors.number_of_reactions,
            1 / reaction_tensors.average_relative_absolute_energies,
            mean_reaction_energy,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test D3 parameters in one script.")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--load", type=str, default="", help="Dataset suffix tag")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="gmtkn-def2")
    parser.add_argument("--basis", type=str, default="def2-QZVPP")
    parser.add_argument("--damping", type=str, choices=["bj", "zero"], default="bj")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--output-column", type=str, default="modified_ai_d3bj")
    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    data_path = VALIDATE_DIR / f"ccdft_{args.basis}_{args.load}_dft-fitset-def2.csv"
    save_para_path = VALIDATE_DIR / f"ccdft_{args.basis}_{args.load}_dft-fitset-def2.json"
    dataset_json_path = DATASET_JSON_DIR / "dft-fitset-def2.json"

    with open(SUBSET_JSON_PATH, "r", encoding="utf-8") as f:
        batch_subset = flatten_subset(json.load(f)["dft-fitset-def2"])

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    data = pd.read_csv(data_path)
    data_name_list = (data["name"].str.split("_def2").str[0]).to_numpy()
    data_dft_ene_kcalmol = torch.tensor(data["scf_ene"].to_numpy() * AU2KCALMOL, device=args.device, dtype=DTYPE)

    model = D3Model(device=args.device, damping=args.damping)
    input_batch = model.obtain_batch_dicts(build_atoms(data_name_list, dataset_json))
    reaction_tensors = build_reaction_tensors(batch_subset, dataset_json, data_name_list, args.device)

    reaction_energy_dft = torch.einsum("ji,i->j", reaction_tensors.molecules_to_reactions, data_dft_ene_kcalmol)
    print(torch.mean(torch.abs(reaction_tensors.reference_energy - reaction_energy_dft)).item(), flush=True)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3200, T_mult=2, eta_min=1e-8
    )

    best = {}
    print(
        f"Epoch\t: Loss, Para={list(model.params.keys())}, Lr={optimizer.param_groups[0]['lr']:.2e}",
        flush=True,
    )

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(model, input_batch, reaction_tensors, data_dft_ene_kcalmol)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            loss_value = float(loss.item())
            params = model.current_params()
            print(
                f"Epoch {epoch}: Loss={loss_value:.6f}, Para={list(params.values())}, "
                f"Lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            if ("loss" not in best) or (loss_value < best["loss"]):
                best = {"epoch": epoch, "loss": loss_value, "parameters": params}

    with open(save_para_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=4)


def run_test(args: argparse.Namespace) -> None:
    data_path = VALIDATE_DIR / f"ccdft_{args.basis}_{args.load}_gmtkn-def2.csv"
    load_para_path = VALIDATE_DIR / f"ccdft_{args.basis}_{args.load}_dft-fitset-def2.json"
    dataset_json_path = DATASET_JSON_DIR / "gmtkn-def2.json"

    with open(SUBSET_JSON_PATH, "r", encoding="utf-8") as f:
        batch_subset = flatten_subset(json.load(f)["full_subset_dict"])

    with open(load_para_path, "r", encoding="utf-8") as f:
        initial_params = json.load(f)["parameters"]

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    model = D3Model(device=args.device, damping=args.damping, initial_params=initial_params)

    data = pd.read_csv(data_path)
    data_name_list = (data["name"].str.split("_def2").str[0]).to_numpy()
    scf_ene_au = data["scf_ene"].to_numpy()
    data_dft_ene_kcalmol = torch.tensor(scf_ene_au * AU2KCALMOL, device=args.device, dtype=DTYPE)

    input_batch = model.obtain_batch_dicts(build_atoms(data_name_list, dataset_json))
    reaction_tensors = build_reaction_tensors(batch_subset, dataset_json, data_name_list, args.device)

    reaction_energy_dft = torch.einsum("ji,i->j", reaction_tensors.molecules_to_reactions, data_dft_ene_kcalmol)
    mean_absolute_deviation = 56.84 / 1505
    mean_reaction_energy = torch.einsum(
        "i,ij,j->j",
        torch.abs(reaction_tensors.reference_energy - reaction_energy_dft),
        reaction_tensors.reactions_to_subset,
        1 / reaction_tensors.number_of_reactions,
    )
    base_score = mean_absolute_deviation * torch.einsum(
        "i,i,i->",
        reaction_tensors.number_of_reactions,
        1 / reaction_tensors.average_relative_absolute_energies,
        mean_reaction_energy,
    )
    print(base_score, flush=True)

    loss = eval_loss_weighted(model, input_batch, reaction_tensors, data_dft_ene_kcalmol)
    print(loss.item(), flush=True)

    data[args.output_column] = model.total_energy_hartree(input_batch, scf_ene_au)
    data.to_csv(data_path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_printoptions(precision=5, sci_mode=False)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
