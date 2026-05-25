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
from torch_dftd.dftd3_xc_params import get_dftd3_default_params

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
    one_over_number_of_reactions: torch.Tensor
    average_relative_absolute_energies: torch.Tensor
    one_over_mae: torch.Tensor


class D3Model(nn.Module):
    def __init__(
        self,
        device: str,
        damping: str,
        initial_params: Optional[Dict[str, float]] = None,
    ):
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

        trainable_params = {}
        const_params = {}
        for param_name in ["s6", "rs6", "s18", "rs18", "alp"]:
            if initial_params.get(param_name, None) is None:
                trainable_params[param_name] = get_dftd3_default_params(
                    damping, xc="b3-lyp", old=False
                )[param_name] * (2 * np.random.rand())
            else:
                if initial_params[param_name] < -999:
                    const_params[param_name] = get_dftd3_default_params(
                        damping, xc="b3-lyp", old=False
                    )[param_name]
                else:
                    const_params[param_name] = initial_params[param_name]

        print(f"Initial trainable D3 parameters: {trainable_params}")
        print(f"Initial constant D3 parameters: {const_params}")

        self.trainable_params = nn.ParameterDict(
            {
                param_name: nn.Parameter(
                    torch.tensor(param_value, dtype=DTYPE, device=device)
                )
                for param_name, param_value in trainable_params.items()
            }
        )
        self.params = {}
        for param_name in trainable_params.keys():
            self.params[param_name] = self.trainable_params[param_name]
        for param_name in const_params.keys():
            self.params[param_name] = const_params[param_name]
        # sort it back to ["s6", "rs6", "s18", "rs18", "alp"]
        self.params = {k: self.params[k] for k in ["s6", "rs6", "s18", "rs18", "alp"]}

        self.calc.dftd_module.params = self.params

    def current_params(self) -> Dict[str, float]:
        out = {}
        for k, v in self.params.items():
            out[k] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
        return out

    def dispersion_energy_kcalmol(
        self, batch_dicts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        self.calc.reset()
        e_disp = self.calc.dftd_module.calc_energy_batch(
            **batch_dicts, damping=self.damping
        )
        return e_disp * units.mol / units.kcal

    def total_energy_hartree(
        self, batch_dicts: Dict[str, torch.Tensor], scf_ene_au: np.ndarray
    ) -> np.ndarray:
        self.calc.reset()
        e_disp = self.calc.dftd_module.calc_energy_batch(
            **batch_dicts, damping=self.damping
        )
        return e_disp.detach().cpu().numpy() / units.Hartree + scf_ene_au

    def obtain_batch_dicts(
        self, atoms_list: Iterable[Atoms]
    ) -> Dict[str, torch.Tensor]:
        input_dicts_list = [self.calc._preprocess_atoms(atoms) for atoms in atoms_list]
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

        batch_dicts = {
            "Z": torch.cat([d["Z"] for d in input_dicts_list], dim=0),
            "pos": torch.cat([d["pos"] for d in input_dicts_list], dim=0),
            "cell": cell_batch,
            "pbc": torch.stack([d["pbc"] for d in input_dicts_list]),
            "shift_pos": torch.cat([d["shift_pos"] for d in input_dicts_list], dim=0),
            "edge_index": torch.cat(
                [
                    d["edge_index"] + shift_index_array[i]
                    for i, d in enumerate(input_dicts_list)
                ],
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

                if molecule_name in dataset_json and isinstance(
                    dataset_json[molecule_name], str
                ):
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

    reference_energy_t = torch.tensor(
        np.array(reference_energy), device=device, dtype=DTYPE
    )
    molecules_to_reactions_t = torch.tensor(
        np.array(molecules_to_reactions), device=device, dtype=DTYPE
    )
    reactions_to_subset_t = torch.tensor(
        np.array(reactions_to_subset), device=device, dtype=DTYPE
    )

    number_of_reactions = torch.sum(reactions_to_subset_t, axis=0)
    one_over_number_of_reactions = torch.where(
        number_of_reactions > 0,
        1 / number_of_reactions,
        torch.zeros_like(number_of_reactions),
    )
    average_relative_absolute_energies = torch.einsum(
        "i,ij,j->j",
        torch.abs(reference_energy_t),
        reactions_to_subset_t,
        one_over_number_of_reactions,
    )
    one_over_mae = torch.where(
        average_relative_absolute_energies > 0,
        1 / average_relative_absolute_energies,
        torch.zeros_like(average_relative_absolute_energies),
    )
    return ReactionTensors(
        reference_energy=reference_energy_t,
        molecules_to_reactions=molecules_to_reactions_t,
        reactions_to_subset=reactions_to_subset_t,
        number_of_reactions=number_of_reactions,
        one_over_number_of_reactions=one_over_number_of_reactions,
        average_relative_absolute_energies=average_relative_absolute_energies,
        one_over_mae=one_over_mae,
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


def reaction_residuals(
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
    return reaction_tensors.reference_energy - reaction_energy


def _trainable_param_names(model: D3Model) -> List[str]:
    return [
        k
        for k, v in model.params.items()
        if isinstance(v, torch.Tensor) and v.requires_grad
    ]


def _get_param_vector(model: D3Model, names: List[str]) -> torch.Tensor:
    if len(names) == 0:
        return torch.empty(0, dtype=DTYPE)
    return torch.stack([model.params[n].detach().clone() for n in names])


def _set_param_vector(model: D3Model, names: List[str], values: torch.Tensor) -> None:
    with torch.no_grad():
        for i, n in enumerate(names):
            model.params[n].copy_(values[i])


def train_with_lm(
    model: D3Model,
    input_batch: Dict[str, torch.Tensor],
    reaction_tensors: ReactionTensors,
    data_dft_ene_kcalmol: torch.Tensor,
    epochs: int,
    print_step: int,
) -> Dict[str, float]:
    names = _trainable_param_names(model)
    if len(names) == 0:
        loss_value = float(
            train_loss(
                model, input_batch, reaction_tensors, data_dft_ene_kcalmol
            ).item()
        )
        return {"epoch": 0, "loss": loss_value, "parameters": model.current_params()}

    lam = 1e-4
    lam_min = 1e-12
    lam_max = 1e8
    lm_eps = 1e-5
    max_trials = 8
    max_step_norm = 1.0
    best = {}

    print(
        f"Epoch\t: Loss, Para={list(model.params.keys())}, Optimizer=LM",
        flush=True,
    )

    for epoch in range(epochs + 1):
        theta = _get_param_vector(model, names)
        r = reaction_residuals(
            model, input_batch, reaction_tensors, data_dft_ene_kcalmol
        ).detach()
        obj = 0.5 * torch.sum(r * r)

        m = r.shape[0]
        n = theta.shape[0]
        jac = torch.empty((m, n), device=theta.device, dtype=theta.dtype)

        for i in range(n):
            step = lm_eps * (1.0 + torch.abs(theta[i]))
            theta_plus = theta.clone()
            theta_minus = theta.clone()
            theta_plus[i] = theta_plus[i] + step
            theta_minus[i] = theta_minus[i] - step
            _set_param_vector(model, names, theta_plus)
            r_plus = reaction_residuals(
                model, input_batch, reaction_tensors, data_dft_ene_kcalmol
            ).detach()
            _set_param_vector(model, names, theta_minus)
            r_minus = reaction_residuals(
                model, input_batch, reaction_tensors, data_dft_ene_kcalmol
            ).detach()
            jac[:, i] = (r_plus - r_minus) / (2.0 * step)

        _set_param_vector(model, names, theta)
        a = jac.T @ jac
        g = jac.T @ r
        d = torch.clamp(torch.diag(a), min=1e-12)

        accepted = False
        rho = float("nan")
        nu = 2.0

        for _ in range(max_trials):
            h = a + lam * torch.diag(d)
            try:
                delta = torch.linalg.solve(h, g)
            except RuntimeError:
                lam = min(lam * nu, lam_max)
                nu *= 2.0
                continue

            delta_norm = torch.linalg.norm(delta)
            if torch.isfinite(delta_norm) and delta_norm > max_step_norm:
                delta = delta * (max_step_norm / (delta_norm + 1e-12))

            theta_new = theta - delta
            _set_param_vector(model, names, theta_new)
            r_new = reaction_residuals(
                model, input_batch, reaction_tensors, data_dft_ene_kcalmol
            ).detach()
            obj_new = 0.5 * torch.sum(r_new * r_new)

            pred = 0.5 * torch.dot(delta, (lam * d * delta + g))
            if (not torch.isfinite(obj_new)) or (not torch.isfinite(pred)) or pred <= 0:
                _set_param_vector(model, names, theta)
                lam = min(lam * nu, lam_max)
                nu *= 2.0
                continue

            rho = float((obj - obj_new) / pred)
            if rho > 0:
                accepted = True
                _set_param_vector(model, names, theta_new)
                lam_factor = max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                lam = max(lam * lam_factor, lam_min)
                break

            _set_param_vector(model, names, theta)
            lam = min(lam * nu, lam_max)
            nu *= 2.0

        if not accepted:
            _set_param_vector(model, names, theta)

        if epoch % print_step == 0:
            loss_value = float(
                train_loss(
                    model, input_batch, reaction_tensors, data_dft_ene_kcalmol
                ).item()
            )
            params = model.current_params()
            print(
                f"Epoch {epoch}: Loss={loss_value:.6f}, Para={[f'{v:.2f}' for v in params.values()]}, "
                f"Lambda={lam:.2e}, Rho={rho:.2e}, Accept={accepted}",
                flush=True,
            )
            if ("loss" not in best) or (loss_value < best["loss"]):
                best = {"epoch": epoch, "loss": loss_value, "parameters": params}

    return best


def train_with_nelder_mead(
    model: D3Model,
    input_batch: Dict[str, torch.Tensor],
    reaction_tensors: ReactionTensors,
    data_dft_ene_kcalmol: torch.Tensor,
    epochs: int,
    print_step: int,
) -> Dict[str, float]:
    names = _trainable_param_names(model)
    if len(names) == 0:
        loss_value = float(
            train_loss(
                model, input_batch, reaction_tensors, data_dft_ene_kcalmol
            ).item()
        )
        return {"epoch": 0, "loss": loss_value, "parameters": model.current_params()}

    x0 = _get_param_vector(model, names)
    n = x0.numel()
    step = 5e-2

    def f(x: torch.Tensor) -> float:
        _set_param_vector(model, names, x)
        with torch.no_grad():
            return float(
                train_loss(
                    model, input_batch, reaction_tensors, data_dft_ene_kcalmol
                ).item()
            )

    simplex = [x0]
    for i in range(n):
        xi = x0.clone()
        xi[i] = xi[i] + step * (1.0 + torch.abs(x0[i]))
        simplex.append(xi)
    fvals = [f(x) for x in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    best = {}

    print(
        f"Epoch\t: Loss, Para={list(model.params.keys())}, Optimizer=Nelder-Mead",
        flush=True,
    )

    for epoch in range(epochs + 1):
        order = sorted(range(len(simplex)), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if epoch % print_step == 0:
            _set_param_vector(model, names, simplex[0])
            loss_value = fvals[0]
            params = model.current_params()
            print(
                f"Epoch {epoch}: Loss={loss_value:.6f}, Para={[f'{v:.2f}' for v in params.values()]}",
                flush=True,
            )
            if ("loss" not in best) or (loss_value < best["loss"]):
                best = {"epoch": epoch, "loss": loss_value, "parameters": params}

        centroid = torch.stack(simplex[:-1], dim=0).mean(dim=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)

        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                best_x = simplex[0]
                simplex = [best_x] + [
                    best_x + sigma * (simplex[i] - best_x)
                    for i in range(1, len(simplex))
                ]
                fvals = [f(x) for x in simplex]

    order = sorted(range(len(simplex)), key=lambda i: fvals[i])
    _set_param_vector(model, names, simplex[order[0]])
    return best


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
        reaction_tensors.one_over_number_of_reactions,
    )
    return torch.abs(
        mean_absolute_deviation
        * torch.einsum(
            "i,i,i->",
            reaction_tensors.number_of_reactions,
            reaction_tensors.one_over_mae,
            mean_reaction_energy,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test D3 parameters in one script."
    )
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument(
        "--load",
        type=str,
        nargs="?",
        const="",
        default="",
        help="Model name to load for training continuation or testing.",
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="gmtkn-def2")
    parser.add_argument("--basis", type=str, default="def2-QZVPP")
    parser.add_argument("--damping", type=str, choices=["bj", "zero"], default="bj")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adagrad", "levenberg-marquardt", "nelder-mead"],
        default="adagrad",
    )
    parser.add_argument("--output-column", type=str, default="modified_ai_d3bj")
    parser.add_argument(
        "--s6",
        type=float,
        default=None,
        help="Pass a value to make it a constant parameter, or pass a negative value to use the default value as a constant parameter. If not passed, it will be a trainable parameter.",
    )
    parser.add_argument(
        "--rs6",
        type=float,
        default=None,
        help="Pass a value to make it a constant parameter, or pass a negative value to use the default value as a constant parameter. If not passed, it will be a trainable parameter.",
    )
    parser.add_argument(
        "--s18",
        type=float,
        default=None,
        help="Pass a value to make it a constant parameter, or pass a negative value to use the default value as a constant parameter. If not passed, it will be a trainable parameter.",
    )
    parser.add_argument(
        "--rs18",
        type=float,
        default=None,
        help="Pass a value to make it a constant parameter, or pass a negative value to use the default value as a constant parameter. If not passed, it will be a trainable parameter.",
    )
    parser.add_argument(
        "--alp",
        type=float,
        default=None,
        help="Pass a value to make it a constant parameter, or pass a negative value to use the default value as a constant parameter. If not passed, it will be a trainable parameter.",
    )
    parser.add_argument("--print_step", type=int, default=100)
    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    data_path = VALIDATE_DIR / f"ccdft_{args.basis}_{args.load}_dft-fitset-def2.csv"
    save_para_path = (
        VALIDATE_DIR
        / f"ccdft_{args.basis}_{args.load}_{args.damping}_dft-fitset-def2.json"
    )
    dataset_json_path = DATASET_JSON_DIR / "dft-fitset-def2.json"

    with open(SUBSET_JSON_PATH, "r", encoding="utf-8") as f:
        batch_subset = flatten_subset(json.load(f)["dft-fitset-def2"])

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    data = pd.read_csv(data_path)
    data_name_list = (data["name"].str.split("_def2").str[0]).to_numpy()
    data_dft_ene_kcalmol = torch.tensor(
        data["scf_ene"].to_numpy() * AU2KCALMOL, device=args.device, dtype=DTYPE
    )

    model = D3Model(
        device=args.device,
        damping=args.damping,
        initial_params={
            "s6": args.s6,
            "rs6": args.rs6,
            "s18": args.s18,
            "rs18": args.rs18,
            "alp": args.alp,
        },
    )
    input_batch = model.obtain_batch_dicts(build_atoms(data_name_list, dataset_json))
    reaction_tensors = build_reaction_tensors(
        batch_subset, dataset_json, data_name_list, args.device
    )

    if args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=args.lr, weight_decay=1e-12
        )
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
            loss = train_loss(
                model, input_batch, reaction_tensors, data_dft_ene_kcalmol
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % args.print_step == 0:
                loss_value = float(loss.item())
                params = model.current_params()
                print(
                    f"Epoch {epoch}: Loss={loss_value:.6f}, Para={[f'{v:.2f}' for v in params.values()]}, "
                    f"Lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
                if ("loss" not in best) or (loss_value < best["loss"]):
                    best = {"epoch": epoch, "loss": loss_value, "parameters": params}
    elif args.optimizer == "levenberg-marquardt":
        best = train_with_lm(
            model,
            input_batch,
            reaction_tensors,
            data_dft_ene_kcalmol,
            args.epochs,
            args.print_step,
        )
    elif args.optimizer == "nelder-mead":
        best = train_with_nelder_mead(
            model,
            input_batch,
            reaction_tensors,
            data_dft_ene_kcalmol,
            args.epochs,
            args.print_step,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    with open(save_para_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=4)


def run_test(args: argparse.Namespace) -> None:
    data_path = VALIDATE_DIR / f"ccdft_{args.basis}_{args.load}_gmtkn-def2.csv"
    load_para_path = (
        VALIDATE_DIR
        / f"ccdft_{args.basis}_{args.load}_{args.damping}_dft-fitset-def2.json"
    )
    dataset_json_path = DATASET_JSON_DIR / "gmtkn-def2.json"

    with open(SUBSET_JSON_PATH, "r", encoding="utf-8") as f:
        batch_subset = flatten_subset(json.load(f)["full_subset_dict"])

    with open(load_para_path, "r", encoding="utf-8") as f:
        initial_params = json.load(f)["parameters"]
    print(f"Loaded parameters for testing: {initial_params}")

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    model = D3Model(
        device=args.device, damping=args.damping, initial_params=initial_params
    )

    data = pd.read_csv(data_path)
    data_name_list = (data["name"].str.split("_def2").str[0]).to_numpy()
    scf_ene_au = data["scf_ene"].to_numpy()
    data_dft_ene_kcalmol = torch.tensor(
        scf_ene_au * AU2KCALMOL, device=args.device, dtype=DTYPE
    )

    input_batch = model.obtain_batch_dicts(build_atoms(data_name_list, dataset_json))
    reaction_tensors = build_reaction_tensors(
        batch_subset, dataset_json, data_name_list, args.device
    )

    reaction_energy_dft = torch.einsum(
        "ji,i->j", reaction_tensors.molecules_to_reactions, data_dft_ene_kcalmol
    )
    mean_absolute_deviation = 56.84 / 1505
    mean_reaction_energy = torch.einsum(
        "i,ij,j->j",
        torch.abs(reaction_tensors.reference_energy - reaction_energy_dft),
        reaction_tensors.reactions_to_subset,
        reaction_tensors.one_over_number_of_reactions,
    )
    base_score = mean_absolute_deviation * torch.einsum(
        "i,i,i->",
        reaction_tensors.number_of_reactions,
        reaction_tensors.one_over_mae,
        mean_reaction_energy,
    )
    print(f"base_score: {base_score.item()}", flush=True)

    loss = eval_loss_weighted(
        model, input_batch, reaction_tensors, data_dft_ene_kcalmol
    )
    print(f"After fitting: {loss.item()}", flush=True)

    data[args.output_column] = model.total_energy_hartree(input_batch, scf_ene_au)
    data.to_csv(data_path, index=False)


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=5, sci_mode=False)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
