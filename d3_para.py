import argparse
import json
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
from scipy.optimize import Bounds, minimize
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from torch_dftd.dftd3_xc_params import get_dftd3_default_params

from cc2cc.utils import AU2KCALMOL

DTYPE = torch.float64
DEFAULT_DEVICE = "cuda"
POSITIVE_PARAM_EPS = 0.0
PARAM_NAMES = ("s6", "rs6", "s18", "rs18", "alp")
SUBSET_JSON_PATH = Path("cc2cc/utils/mol_dataset/subset.json")
DATASET_JSON_DIR = Path("cc2cc/utils/mol_dataset")
VALIDATE_DIR = Path("validate_hkqai_done")
SCIPY_MINIMIZE_METHODS = {
    "levenberg-marquardt": ("L-BFGS-B", True),
    "l-bfgs-b": ("L-BFGS-B", True),
    "nelder-mead": ("Nelder-Mead", False),
    "powell": ("Powell", False),
    "slsqp": ("SLSQP", True),
    "tnc": ("TNC", True),
    "trust-constr": ("trust-constr", True),
}
OPTIMIZERS = {"adagrad", *SCIPY_MINIMIZE_METHODS}
OPTIMIZER_ALIASES = {
    "ada": "adagrad",
    "ag": "adagrad",
    "levenberg": "levenberg-marquardt",
    "lm": "levenberg-marquardt",
    "lbfgsb": "l-bfgs-b",
    "lbfgs": "l-bfgs-b",
    "nelder": "nelder-mead",
    "nm": "nelder-mead",
    "pow": "powell",
    "trust": "trust-constr",
    "trust-region": "trust-constr",
    "tr": "trust-constr",
}
IF_PRINT = False


@dataclass
class ReactionTensors:
    reference_energy: torch.Tensor
    molecules_to_reactions: torch.Tensor
    reactions_to_subset: torch.Tensor
    number_of_reactions: torch.Tensor
    one_over_number_of_reactions: torch.Tensor
    average_relative_absolute_energies: torch.Tensor
    one_over_mae: torch.Tensor
    reaction_weight: Optional[torch.Tensor] = None


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
        for param_name in PARAM_NAMES:
            if initial_params.get(param_name, None) is None:
                trainable_params[param_name] = np.random.uniform(0.8, 2.0)
            else:
                if initial_params[param_name] < -999:
                    const_params[param_name] = get_dftd3_default_params(
                        damping, xc="b3-lyp", old=False
                    )[param_name]
                else:
                    const_params[param_name] = initial_params[param_name]

        if IF_PRINT:
            print(f"Initial trainable D3 parameters: {trainable_params}", flush=True)
            print(f"Initial constant D3 parameters: {const_params}", flush=True)

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
        self.params = {k: self.params[k] for k in PARAM_NAMES}

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
    name_subset_weight_dict: Optional[Dict[str, float]] = None,
) -> ReactionTensors:
    reference_energy = []
    molecules_to_reactions = []
    reactions_to_subset = []
    if name_subset_weight_dict is not None:
        reaction_weight = []
    else:
        reaction_weight = None

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
                if name_subset_weight_dict is not None:
                    reaction_weight.append(
                        name_subset_weight_dict.get(subset_name, 1.0)
                    )

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
        reaction_weight=(
            torch.tensor(reaction_weight, device=device, dtype=DTYPE)
            if reaction_weight
            else None
        ),
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
    if reaction_tensors.reaction_weight is not None:
        reaction_energy = reaction_energy * reaction_tensors.reaction_weight
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


def _set_param_vector(model: D3Model, names: List[str], values: torch.Tensor) -> None:
    with torch.no_grad():
        for i, n in enumerate(names):
            model.params[n].copy_(values[i])


def _update_best(
    best: Dict[str, float], epoch: int, loss_value: float, params: Dict[str, float]
) -> None:
    if ("loss" not in best) or (loss_value < best["loss"]):
        best.update({"epoch": epoch, "loss": loss_value, "parameters": params})


def normalize_optimizer(value: str) -> str:
    key = value.lower().replace("_", "-")
    if key in OPTIMIZERS:
        return key
    if key not in OPTIMIZER_ALIASES:
        choices = ", ".join(sorted(OPTIMIZERS | set(OPTIMIZER_ALIASES)))
        raise argparse.ArgumentTypeError(
            f"unknown optimizer '{value}'. Available optimizers/aliases: {choices}"
        )
    return OPTIMIZER_ALIASES[key]


def scipy_minimize_options(optimizer: str, epochs: int) -> Dict:
    max_steps = max(1, epochs)
    if optimizer == "nelder-mead":
        return {
            "maxiter": max_steps,
            "maxfev": max_steps,
            "xatol": 0.0,
            "fatol": 0.0,
            "adaptive": True,
        }
    if optimizer == "powell":
        return {"maxiter": max_steps, "maxfev": max_steps, "xtol": 0.0, "ftol": 0.0}
    if optimizer == "tnc":
        return {"maxfun": max_steps}
    if optimizer == "trust-constr":
        return {
            "maxiter": max_steps,
            "gtol": 1e-8,
            "xtol": 1e-8,
            "barrier_tol": 1e-8,
        }
    if optimizer in {"levenberg-marquardt", "l-bfgs-b"}:
        return {"maxiter": max_steps, "maxfun": max_steps}
    return {"maxiter": max_steps}


def _train_with_scipy_minimize(
    model: D3Model,
    input_batch: Dict[str, torch.Tensor],
    reaction_tensors: ReactionTensors,
    data_dft_ene_kcalmol: torch.Tensor,
    print_step: int,
    method: str,
    optimizer_label: str,
    use_jac: bool,
    options: Dict,
) -> Dict[str, float]:
    names = list(model.trainable_params.keys())

    best = {}
    state = {"nfev": 0}
    x0_t = torch.stack([model.params[n].detach().clone() for n in names])
    x0 = x0_t.detach().cpu().numpy()
    bounds = Bounds(
        lb=np.full(len(names), POSITIVE_PARAM_EPS),
        ub=np.full(len(names), np.inf),
    )
    cache = {"x": None, "grad": None, "obj": 0.0, "train_loss": 0.0}

    if IF_PRINT:
        print(
            f"Epoch\t: Loss, Para={list(model.params.keys())}, Optimizer={optimizer_label}",
            flush=True,
        )

    def _objective_and_grad(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        if cache["x"] is not None and np.array_equal(cache["x"], x_arr):
            return cache["obj"], cache["grad"], cache["train_loss"]

        theta_t = torch.tensor(x_arr, dtype=DTYPE, device=x0_t.device)
        _set_param_vector(model, names, theta_t)
        r = reaction_residuals(
            model, input_batch, reaction_tensors, data_dft_ene_kcalmol
        )
        train_loss_value = torch.einsum(
            "i,ij,j,j->j",
            torch.abs(r),
            reaction_tensors.reactions_to_subset,
            reaction_tensors.one_over_mae,
            reaction_tensors.one_over_number_of_reactions,
        )
        train_loss_value = float(torch.mean(torch.abs(train_loss_value)).item())
        if reaction_tensors.reaction_weight is not None:
            r = r * reaction_tensors.reaction_weight
        obj_t = torch.einsum(
            "i,ij,j,j->j",
            torch.abs(r),
            reaction_tensors.reactions_to_subset,
            reaction_tensors.one_over_mae,
            reaction_tensors.one_over_number_of_reactions,
        )
        obj_t = torch.mean(torch.abs(obj_t))
        obj_value = float(obj_t.item())
        grads = torch.autograd.grad(
            obj_t,
            [model.params[n] for n in names],
            retain_graph=False,
            create_graph=False,
        )
        grad = np.array(
            [float(g.detach().item()) for g in grads],
            dtype=np.float64,
        )
        cache["x"] = x_arr.copy()
        cache["grad"] = grad
        cache["obj"] = obj_value
        cache["train_loss"] = train_loss_value
        return obj_value, grad, train_loss_value

    def objective_fn(x: np.ndarray) -> float:
        obj_value, _, train_loss_value = _objective_and_grad(x)

        epoch = state["nfev"]
        if epoch % print_step == 0:
            params = model.current_params()
            if IF_PRINT:
                print(
                    f"Epoch {epoch}: Loss={train_loss_value:.6f}, Para={[f'{v:.2f}' for v in params.values()]}",
                    flush=True,
                )
        _update_best(best, epoch, train_loss_value, model.current_params())
        state["nfev"] += 1
        return obj_value

    def jac_fn(x: np.ndarray) -> np.ndarray:
        _, grad, _ = _objective_and_grad(x)
        return grad

    result = minimize(
        objective_fn,
        x0,
        method=method,
        jac=jac_fn if use_jac else None,
        bounds=bounds,
        options=options,
    )
    theta_opt_t = torch.tensor(result.x, dtype=DTYPE, device=x0_t.device)
    _set_param_vector(model, names, theta_opt_t)
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
        type=normalize_optimizer,
        default="adagrad",
        metavar="OPT",
        help=(
            "Optimizer name or alias. Examples: adagrad/ada, "
            "levenberg-marquardt/lm, nelder-mead/nm, lbfgs, powell, slsqp, "
            "tnc, trust-constr/tr."
        ),
    )
    param_help = (
        "Pass a value to make it a constant parameter, or pass a negative value to "
        "use the default value as a constant parameter. If not passed, it will be a "
        "trainable parameter."
    )
    for param_name in PARAM_NAMES:
        parser.add_argument(
            f"--{param_name}", type=float, default=None, help=param_help
        )
    parser.add_argument("--print_step", type=int, default=5000)
    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    data_path = VALIDATE_DIR / f"ccdft_{args.basis}_{args.load}_dft-fitset-def2.csv"
    save_para_path = (
        VALIDATE_DIR
        / f"ccdft_{args.basis}_{args.load}_{args.damping}_dft-fitset-def2.json"
    )
    dataset_json_path = DATASET_JSON_DIR / "dft-fitset-def2.json"

    with open(SUBSET_JSON_PATH, "r", encoding="utf-8") as f:
        json_file = json.load(f)
        batch_subset = flatten_subset(json_file["dft-fitset-def2"])
        if "dft-fitset-def2-weight" in json_file:
            name_subset_weight_dict = json_file["dft-fitset-def2-weight"]
        else:
            name_subset_weight_dict = None

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
        initial_params={name: getattr(args, name) for name in PARAM_NAMES},
    )
    input_batch = model.obtain_batch_dicts(build_atoms(data_name_list, dataset_json))
    reaction_tensors = build_reaction_tensors(
        batch_subset,
        dataset_json,
        data_name_list,
        args.device,
        name_subset_weight_dict,
    )

    if args.optimizer in SCIPY_MINIMIZE_METHODS:
        method, use_jac = SCIPY_MINIMIZE_METHODS[args.optimizer]
        optimizer_label = f"{args.optimizer.upper()} (SciPy minimize)"
        best = _train_with_scipy_minimize(
            model=model,
            input_batch=input_batch,
            reaction_tensors=reaction_tensors,
            data_dft_ene_kcalmol=data_dft_ene_kcalmol,
            print_step=args.print_step,
            method=method,
            optimizer_label=optimizer_label,
            use_jac=use_jac,
            options=scipy_minimize_options(args.optimizer, args.epochs),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    with open(save_para_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=4)

    # save the final data after training for testing
    data[f"modified_ai_d3{args.damping}"] = model.total_energy_hartree(
        input_batch, data["scf_ene"].to_numpy()
    )
    data.to_csv(data_path, index=False)


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
        batch_subset,
        dataset_json,
        data_name_list,
        args.device,
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

    data[f"modified_ai_d3{args.damping}"] = model.total_energy_hartree(
        input_batch, scf_ene_au
    )
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
