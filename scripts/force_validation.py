"""Build displacement geometries and compare model-SCF gradients."""

import argparse
import copy
import csv
import json
import re
from pathlib import Path

import numpy as np
from pyscf.lib import param


STEPS_BOHR = (0.004, 0.002, 0.001)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
GRADIENT_ROW = rf"\[\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\]"


def build_dataset(source):
    legacy = source["molecule_Force"]
    names = list(legacy)
    result = {
        "molecule": names,
        "spin": {name: source["spin"][name] for name in legacy},
        "charge": {name: source["charge"][name] for name in legacy},
        "molecule_Force_legacy": list(legacy),
        **{name: copy.deepcopy(source[name]) for name in legacy},
    }
    cases = {}
    for label, original in (
        ("ch4", "Force-ch4-1.000"),
        ("h2o", "W4_11-h2o"),
        ("oh", "W4_11-oh"),
        ("hal59-brbr_fcch", "HAL59-BrBr_FCCH"),
        ("w4_11-sif4", "W4_11-sif4"),
    ):
        center = f"Force-{label}-center"
        atoms = copy.deepcopy(source[original])
        offsets = [[1, 0.13, 0.08, 0.04]] if label == "h2o" else []
        for atom, *offset in offsets:
            atoms[atom][1:] = [x + dx for x, dx in zip(atoms[atom][1:], offset)]
        result[center] = atoms
        names.append(center)
        case = dict(center=center, source=original, offsets_angstrom=offsets, pairs={})
        for step in STEPS_BOHR:
            pairs = []
            for atom in range(len(atoms)):
                for axis in range(3):
                    pair = dict(atom=atom, axis=axis)
                    for sign, suffix in ((-1, "minus"), (1, "plus")):
                        name = f"Force-{label}-a{atom}-{'xyz'[axis]}-h{step:.3f}-{suffix}"
                        displaced = copy.deepcopy(atoms)
                        displaced[atom][axis + 1] += sign * step * param.BOHR
                        result[name] = displaced
                        names.append(name)
                        pair[suffix] = name
                    pairs.append(pair)
            case["pairs"][f"{step:.3f}"] = pairs
        cases[label] = case
        for name in names:
            if name.startswith(f"Force-{label}-") and name not in legacy:
                result["spin"][name] = source["spin"][original]
                result["charge"][name] = source["charge"][original]
    result.update({
        "molecule_Force": list(names),
        "molecule_Force_centers": [case["center"] for case in cases.values()],
        "reaction-Force": {},
        "molecule1-Force": list(names),
        "force_validation": dict(steps_bohr=list(STEPS_BOHR), cases=cases),
    })
    return result


def read_force_csv(paths, dataset, basis, completed_only=False, *, settings=None):
    """Read uncorrected SCF energies and gradients from validation CSVs.

    CSVs do not record calculation settings. By default, records from different
    sources cannot be compared; explicitly supplied settings are caller-verified.
    """
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("At least one CSV path is required")
    if settings is not None:
        if settings.get("basis") != basis:
            raise ValueError(f"CSV settings must specify basis={basis}")
        settings = dict(settings)

    names = set(dataset["molecule"])
    records = {}
    matched = 0
    for path in paths:
        record_settings = settings
        if record_settings is None:
            record_settings = dict(basis=basis, csv_source=str(path.resolve()))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = reader.fieldnames or []
            for column in ("name", "scf_ene", "converged", "scf_grad"):
                count = fields.count(column)
                if count > 1 or (column != "scf_grad" and count != 1):
                    raise ValueError(f"{path}: invalid or missing {column!r} column")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}:{reader.line_num}: invalid column count")
                run = row["name"].strip()
                if not run.endswith("_" + basis):
                    continue
                name = run.removesuffix("_" + basis)
                if name not in names:
                    continue
                matched += 1
                converged = row["converged"].strip().lower()
                if converged not in ("true", "false", ""):
                    raise ValueError(f"{path}:{reader.line_num}: invalid convergence status")
                if converged != "true":
                    if completed_only:
                        continue
                    raise ValueError(f"{name}: SCF is not converged")
                if name in records:
                    raise ValueError(f"Duplicate run: {name}")
                try:
                    energy = float(row["scf_ene"])
                except ValueError as exc:
                    raise ValueError(f"{name}: invalid SCF energy") from exc
                if not np.isfinite(energy):
                    raise ValueError(f"{name}: invalid SCF energy")
                gradient = None
                value = row.get("scf_grad", "").strip()
                if value:
                    if re.fullmatch(rf"\s*\[\s*(?:{GRADIENT_ROW}\s*)+\]\s*", value) is None:
                        raise ValueError(f"{name}: malformed SCF gradient")
                    gradient = np.array(re.findall(GRADIENT_ROW, value), dtype=np.float64)
                    if gradient.shape != (len(dataset[name]), 3):
                        raise ValueError(f"{name}: incorrect SCF gradient shape {gradient.shape}")
                    if not np.isfinite(gradient).all():
                        raise ValueError(f"{name}: invalid SCF gradient")
                records[name] = dict(energy=energy, gradient=gradient, settings=record_settings)
    if not matched:
        raise ValueError(f"No {basis} CSV records matched the dataset")
    return records


def gradient_report(dataset, energies, gradients_on, gradients_off=None,
                    completed_only=False):
    modes = [("on", gradients_on)]
    if gradients_off is not None:
        modes.append(("off", gradients_off))
    reports = []
    for label, case in dataset["force_validation"]["cases"].items():
        center = case["center"]
        if completed_only:
            if (center not in energies or center not in gradients_on
                    or gradients_on[center]["gradient"] is None):
                continue
            if gradients_off is not None and (
                    center not in gradients_off
                    or gradients_off[center]["gradient"] is None):
                continue
            pairs_by_step = {
                step: pairs for step, pairs in case["pairs"].items()
                if all(pair[side] in energies
                       for pair in pairs for side in ("minus", "plus"))
            }
            if not pairs_by_step:
                continue
        else:
            pairs_by_step = case["pairs"]
        settings = energies[center]["settings"]
        for records in (energies, *(records for _, records in modes)):
            for name, record in records.items():
                if record["settings"] != settings:
                    raise ValueError(f"{name}: inconsistent model/grid/precision settings")
        coords = np.array([a[1:] for a in dataset[center]], dtype=np.float64) / param.BOHR
        invariance = {}
        for mode, records in modes:
            gradient = records[center]["gradient"]
            if gradient is None:
                raise ValueError(f"{center}: missing grid-response-{mode} gradient")
            invariance[mode] = dict(
                net_gradient=gradient.sum(axis=0),
                torque=np.cross(coords - coords.mean(axis=0), gradient).sum(axis=0),
            )
        previous = {}
        for step, pairs in pairs_by_step.items():
            fd = np.empty_like(coords)
            for pair in pairs:
                atom, axis = pair["atom"], pair["axis"]
                offsets = []
                for endpoint in ("minus", "plus"):
                    name = pair[endpoint]
                    geometry = np.array([a[1:] for a in dataset[name]], dtype=np.float64)
                    delta = geometry / param.BOHR - coords
                    if np.argwhere(np.abs(delta) > 1e-12).tolist() != [[atom, axis]]:
                        raise ValueError(f"{name}: incorrect displacement")
                    offsets.append(delta[atom, axis])
                da, db = offsets
                if not np.allclose([da, db], [-float(step), float(step)], rtol=0, atol=1e-12):
                    raise ValueError(f"{label}: incorrect step {step}")
                ea = energies[pair["minus"]]["energy"] - energies[center]["energy"]
                eb = energies[pair["plus"]]["energy"] - energies[center]["energy"]
                fd[atom, axis] = (db * ea / da - da * eb / db) / (db - da)
            if len(pairs) != coords.size or len({(p["atom"], p["axis"]) for p in pairs}) != coords.size:
                raise ValueError(f"{label}: incomplete Cartesian coverage")
            for mode, records in modes:
                error = fd - records[center]["gradient"]
                maximum = np.max(np.abs(error))
                rms = np.sqrt(np.mean(error ** 2))
                order = None
                if mode in previous:
                    old_step, old_rms = previous[mode]
                    if rms > 0 and old_rms > 0:
                        order = np.log(old_rms / rms) / np.log(old_step / float(step))
                previous[mode] = (float(step), rms)
                reports.append(dict(
                    case=label, step_bohr=float(step), grid_response=mode,
                    max_error=maximum, rms_error=rms, observed_order=order,
                    finite_difference=fd.copy(), **invariance[mode],
                ))
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="Source dataset for geometry generation")
    inputs.add_argument("--csv", type=Path, help="Validation CSV file to report")
    parser.add_argument("--output", type=Path, help="Output path for generated geometries")
    parser.add_argument("--dataset", type=Path, help="Force-validation dataset JSON")
    parser.add_argument("--basis", help="Basis suffix used in CSV record names")
    parser.add_argument("--completed-only", action="store_true",
                        help="Report only converged, complete Cartesian steps")
    parser.add_argument("--grid-response", choices=("on", "off", "unknown"), default="unknown",
                        help="Declared gradient setting; CSVs cannot verify it")
    args = parser.parse_args()
    if args.source is not None:
        if args.output is None:
            parser.error("--output is required with --source")
        if (args.dataset is not None or args.basis is not None or args.completed_only
                or args.grid_response != "unknown"):
            parser.error("--dataset, --basis, --completed-only and --grid-response require --csv")
        args.output.write_text(json.dumps(build_dataset(json.loads(args.source.read_text())),
                                          indent=4) + "\n")
    else:
        if args.output is not None:
            parser.error("--output is only valid with --source")
        if args.dataset is None or args.basis is None:
            parser.error("--dataset and --basis are required with --csv")
        dataset = json.loads(args.dataset.read_text())
        records = read_force_csv([args.csv], dataset, args.basis,
                                 completed_only=args.completed_only)
        reports = gradient_report(dataset, records, records,
                                  completed_only=args.completed_only)
        for report in reports:
            report["grid_response"] = args.grid_response
        print(f"Completed runs: {len(records)}; Cartesian reports: {len(reports)}")
        print("Gradient errors: Hartree/Bohr; torque: Hartree")
        print("CSV does not verify model, grid, SCF tolerances, precision, "
              "rotation, geometry/atom order, units, or grid response.")
        columns = (
            ("Case", 6), ("h/Bohr", 8), ("Grid", 9), ("Max error", 13),
            ("RMS error", 13), ("Order", 8), ("Net grad max", 14),
            ("Torque max", 13),
        )
        print(" ".join(f"{name:>{width}}" for name, width in columns))
        print("-" * (sum(width for _, width in columns) + len(columns) - 1))
        for report in reports:
            order = report["observed_order"]
            values = (
                f"{report['case']:<6}",
                f"{report['step_bohr']:8.3f}",
                f"{report['grid_response']:>9}",
                f"{report['max_error']:13.4e}",
                f"{report['rms_error']:13.4e}",
                f"{order:8.3f}" if order is not None else f"{'-':>8}",
                f"{np.max(np.abs(report['net_gradient'])):14.4e}",
                f"{np.max(np.abs(report['torque'])):13.4e}",
            )
            print(" ".join(values))
