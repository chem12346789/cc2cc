"""Build displacement geometries and compare logged model-SCF gradients."""

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
from pyscf.lib import param


STEPS_BOHR = (0.004, 0.002, 0.001)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


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


def read_force_logs(paths, dataset, basis, grid_response, conv_tol=1e-11,
                    conv_tol_grad=1e-7, completed_only=False):
    records = {}
    for path in paths:
        text = Path(path).read_text()
        settings = {}
        for key in ("load", "load_epoch", "grid_level", "basis", "dataset", "precision"):
            match = re.search(rf"^{key}: (.+)$", text, re.M)
            if match is None:
                raise ValueError(f"{path}: missing {key}")
            settings[key] = match.group(1).strip()
        if settings["basis"] != basis or settings["precision"] != "float64":
            raise ValueError(f"{path}: require basis={basis}, precision=float64")
        for flag in ("if_rotate", "if_rotate_random"):
            if re.search(rf"^{flag}: False\s*$", text, re.M) is None:
                raise ValueError(f"{path}: require {flag}=False")
        for block in re.split(r"^RUN: ", text, flags=re.M)[1:]:
            run = block.splitlines()[0].strip()
            if not run.endswith("_" + basis):
                continue
            name = run.removesuffix("_" + basis)
            if name not in dataset["molecule"]:
                continue
            converged = list(re.finditer(
                rf"^converged SCF energy =\s*({NUMBER})", block, re.M
            ))
            if not converged:
                if completed_only:
                    continue
                raise ValueError(f"{name}: SCF is not converged")
            energy = converged[-1]
            if re.search(r"SCF not converged|not converged!!!", block[energy.end():]):
                if completed_only:
                    continue
                raise ValueError(f"{name}: final SCF is not converged")
            if name in records:
                raise ValueError(f"Duplicate run: {name}")
            for key, limit in (("SCF conv_tol", conv_tol),
                               ("SCF conv_tol_grad", conv_tol_grad)):
                match = re.search(rf"^{key} =\s*({NUMBER})", block, re.M)
                if match is None:
                    # GPU4PySCF UKS may omit the per-run SCF settings banner.
                    argument = key.removeprefix("SCF ")
                    match = re.search(rf"^{argument}:\s*({NUMBER})\s*$", text, re.M)
                if match is None or float(match.group(1)) > limit:
                    raise ValueError(f"{name}: require {key} <= {limit}")
            rows = re.findall(
                rf"^\s*(\d+)\s+([A-Z][a-z]?)\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*$",
                block, re.M,
            )
            gradient = None
            if rows:
                atoms = dataset[name]
                if ([int(row[0]) for row in rows] != list(range(len(atoms)))
                        or [row[1] for row in rows] != [atom[0] for atom in atoms]):
                    raise ValueError(f"{name}: incomplete or misordered gradient")
                if ("unit = Eh/Bohr" not in block
                        or f"grid_response = {bool(grid_response)}" not in block):
                    raise ValueError(f"{name}: wrong gradient units/grid-response setting")
                gradient = np.array([row[2:] for row in rows], dtype=np.float64)
            records[name] = dict(energy=float(energy.group(1)), gradient=gradient,
                                 settings=settings)
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_dataset(json.loads(args.source.read_text())),
                                      indent=4) + "\n")
