"""Compare reaction energies on model-SCF and fixed B3LYP densities."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collect_info import AU2KCALMOL, Collect_info


def load_energies(path, basis, columns):
    files = sorted(path.glob("ccdft_*_molecule_*.csv")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No molecular validation CSVs found in {path}")
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        missing = {"name", *columns} - set(frame.columns)
        if missing:
            raise ValueError(f"{file}: missing columns {sorted(missing)}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    suffix = f"_{basis}"
    if not data["name"].str.endswith(suffix).fillna(False).all():
        raise ValueError(f"{path}: molecule names must end with {suffix!r}")
    data["name"] = data["name"].str.removesuffix(suffix)
    if data["name"].duplicated().any():
        names = data.loc[data["name"].duplicated(), "name"].tolist()
        raise ValueError(f"{path}: duplicate molecules: {names}")
    return data.set_index("name")


def build_reactions(dataset):
    reactions = []
    for key, entries in dataset.items():
        if not key.startswith("reaction-"):
            continue
        subset = key.removeprefix("reaction-")
        alias = Collect_info._subset_alias(subset)
        references = [float(entry["reference"]) for entry in entries.values()]
        mean_reference = np.mean(np.abs(references))
        for reaction_id, entry in entries.items():
            coefficients = {}
            for system, coefficient in zip(
                entry["systems"], entry["stoichiometry"], strict=True
            ):
                name = f"{alias}-{system}"
                if isinstance(dataset.get(name), str):
                    name = dataset[name]
                coefficients[name] = coefficients.get(name, 0) + int(coefficient)
            # DIET datasets supply weights; full GMTKN uses 56.84/<|reference|>.
            weight = float(entry["weight"]) if "weight" in entry else 56.84 / mean_reference
            reactions.append((subset, reaction_id, entry["reference"], weight, coefficients))
    if not reactions:
        raise ValueError("Dataset contains no reaction-* entries")
    return reactions


def compare(scf, b3lyp, reactions, column):
    rows = []
    for subset, reaction_id, reference, weight, coefficients in reactions:
        names = list(coefficients)
        stoich = np.array(list(coefficients.values()), dtype=np.float64)
        scf_values = scf.loc[names, column].to_numpy(dtype=np.float64)
        b3_values = b3lyp.loc[names, column].to_numpy(dtype=np.float64)
        scf_energy = np.dot(stoich, scf_values) * AU2KCALMOL
        b3_energy = np.dot(stoich, b3_values) * AU2KCALMOL
        shift = scf_energy - b3_energy
        gross_shift = np.sum(np.abs(stoich * (scf_values - b3_values))) * AU2KCALMOL
        rows.append({
            "subset": subset,
            "reaction": reaction_id,
            "reference": reference,
            "weight": weight,
            "B3_error": b3_energy - reference,
            "SCF_error": scf_energy - reference,
            "shift": shift,
            "gross_shift": gross_shift,
        })
    result = pd.DataFrame(rows)
    result["gain"] = result["B3_error"].abs() - result["SCF_error"].abs()
    result["weighted_gain"] = result["weight"] * result["gain"]
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        raise ValueError(f"{column}: non-finite reaction values or weights")
    return result


def print_report(result, scf, b3lyp, column, top):
    count = len(result)
    b3_abs = result["B3_error"].abs()
    scf_abs = result["SCF_error"].abs()
    abs_shift = result["shift"].abs()
    gain = result["gain"]
    weighted_gain = result["weighted_gain"]
    print(f"\n{'=' * 78}\nEnergy column: {column} | {count} paired reactions")
    print("All energies/errors below are in kcal/mol; shift = SCF - B3LYP-density.")
    print(f"{'Metric':42s} {'B3LYP density':>15s} {'Model SCF':>15s}")
    print(f"{'Reaction MAE':42s} {b3_abs.mean():15.6f} {scf_abs.mean():15.6f}")
    print(f"{'WTMAD-2 (mean of weight * |error|)':42s} "
          f"{(result.weight * b3_abs).mean():15.6f} "
          f"{(result.weight * scf_abs).mean():15.6f}")
    print(f"Paired |reaction shift|: mean={abs_shift.mean():.6f}, "
          f"median={abs_shift.median():.6f}, max={abs_shift.max():.6f}")
    print(f"Weighted mean |reaction shift|: {(result.weight * abs_shift).mean():.6f}")
    print(f"MAE gain (positive = SCF better): {gain.mean():.6f}")
    print(f"WTMAD-2 gain: {weighted_gain.mean():.6f}")
    tied = np.isclose(gain, 0, atol=1e-6, rtol=0)
    print(f"Improved / worsened / tied (1e-6 tolerance): "
          f"{((gain > 0) & ~tied).sum()} / {((gain < 0) & ~tied).sum()} / {tied.sum()}")
    print(f"MAE gains / losses per reaction: "
          f"{gain.clip(lower=0).sum() / count:.6f} / "
          f"{-gain.clip(upper=0).sum() / count:.6f}")
    print(f"WTMAD-2 gains / losses per reaction: "
          f"{weighted_gain.clip(lower=0).sum() / count:.6f} / "
          f"{-weighted_gain.clip(upper=0).sum() / count:.6f}")
    molecular_shift = (scf[column] - b3lyp[column]) * AU2KCALMOL
    print(f"Molecular SCF-B3 shifts: mean={molecular_shift.mean():.6f}, "
          f"mean |shift|={molecular_shift.abs().mean():.6f}, "
          f"range=[{molecular_shift.min():.6f}, {molecular_shift.max():.6f}]")
    gross = result.gross_shift.sum()
    if gross > 0:
        print(f"Within-reaction molecular-shift cancellation: "
              f"{100 * (1 - abs_shift.sum() / gross):.2f}% "
              "(1 - sum|reaction shift| / sum|stoichiometric molecular shifts|)")
    else:
        print("Within-reaction cancellation: undefined (all molecular shifts are zero).")
    print("\nSubset summary (contributions sum to total WTMAD-2 gain):")
    summary = result.assign(B3_MAE=b3_abs, SCF_MAE=scf_abs, paired_shift=abs_shift)
    summary = summary.groupby("subset", sort=True).agg(
        N=("gain", "size"), B3_MAE=("B3_MAE", "mean"),
        SCF_MAE=("SCF_MAE", "mean"), paired_shift=("paired_shift", "mean"),
        WTMAD_gain=("weighted_gain", "sum"),
    )
    summary["WTMAD_gain"] /= count
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    for title, ascending in (("Largest weighted improvements", False),
                             ("Largest weighted deteriorations", True)):
        selected = result.loc[weighted_gain > 0 if not ascending else weighted_gain < 0]
        selected = selected.sort_values("weighted_gain", ascending=ascending).head(top)
        selected = selected.assign(WTMAD_gain=selected.weighted_gain / count)
        print(f"\n{title} (positive gain = SCF better):")
        print(selected[["subset", "reaction", "B3_error", "SCF_error",
                        "shift", "gain", "WTMAD_gain"]].to_string(
                            index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nLargest paired reaction shifts:")
    selected = result.loc[abs_shift.nlargest(top).index]
    print(selected[["subset", "reaction", "B3_error", "SCF_error", "shift"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scf", type=Path, required=True,
                        help="Model-SCF checkpoint directory or aggregated molecular CSV")
    parser.add_argument("--b3lyp", type=Path, required=True,
                        help="Same model evaluated on B3LYP density: directory or CSV")
    parser.add_argument("--basis", default="def2-QZVP(D)")
    parser.add_argument("--dataset", type=Path, default=(
        Path(__file__).resolve().parents[1]
        / "cc2cc/utils/mol_dataset/gmtkn-diet100-def2.json"))
    parser.add_argument("--columns", nargs="+", default=["scf_ene", "scf_d3bj_ene"])
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    with args.dataset.open() as file:
        reactions = build_reactions(json.load(file))
    scf = load_energies(args.scf, args.basis, args.columns)
    b3lyp = load_energies(args.b3lyp, args.basis, args.columns)
    required = sorted({name for *_, coefficients in reactions for name in coefficients})
    for label, data in (("SCF", scf), ("B3LYP density", b3lyp)):
        missing = set(required) - set(data.index)
        if missing:
            raise ValueError(f"{label}: missing {len(missing)} required molecules: {sorted(missing)}")
        if not np.isfinite(data.loc[required, args.columns].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{label}: selected energies contain missing/non-finite values")
        if "converged" in data.columns:
            flags = data.loc[required, "converged"].astype(str).str.lower()
            print(f"{label} convergence: true={flags.eq('true').sum()}, "
                  f"false={flags.eq('false').sum()}, "
                  f"unreported={len(flags) - flags.isin(['true', 'false']).sum()}")
            if flags.eq("false").any():
                print(f"WARNING: {label} includes unconverged energies.")
        else:
            print(f"WARNING: {label} has no convergence metadata.")
    scf, b3lyp = scf.loc[required], b3lyp.loc[required]
    print(f"SCF input: {args.scf}\nB3LYP-density input: {args.b3lyp}")
    print(f"Dataset: {args.dataset}\nCoverage: {len(required)} molecules; "
          f"{len(reactions)} reactions (complete in both inputs).")
    print("Assumption: same checkpoint, geometry, basis, functional and numerical settings; "
          "only density differs.")
    for column in args.columns:
        print_report(compare(scf, b3lyp, reactions, column), scf, b3lyp, column, args.top)
    print("\nInterpretation: compare paired shifts with MAE/WTMAD-2 changes. Similar scores "
          "alone do not imply similar predictions.")
    print("Gains and losses quantify between-reaction averaging; cancellation above is "
          "within reactions, before absolute errors are taken.")
    print("Energy-only results cannot establish which electron density is closer to exact.")


if __name__ == "__main__":
    main()
