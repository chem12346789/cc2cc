"""Check rotational invariance of model-SCF energies and gradient covariance."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc2cc.utils.helper.units import AU2KCALMOL
from scripts.force_validation import read_force_csv


def rotation_matrix(operations):
    matrix = np.eye(3, dtype=np.float64)
    for axis, degrees in operations:
        angle = np.deg2rad(degrees)
        c, s = np.cos(angle), np.sin(angle)
        matrices = {
            "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64),
        }
        if axis not in matrices:
            raise ValueError(f"Unknown rotation axis: {axis}")
        matrix = matrices[axis] @ matrix
    return matrix


def rotation_report(dataset, records, energy_tol=1e-5, gradient_tol=1e-5,
                    completed_only=False, require_gradients=False):
    """Compare converged orientations to their reference geometry and results."""
    for name, tolerance in (("energy", energy_tol), ("gradient", gradient_tol)):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{name} tolerance must be finite and nonnegative")

    metadata = dataset["rotation_validation"]
    orientations = metadata["orientations"]
    if (not metadata["sources"] or orientations.get("reference") != []
            or len(orientations) < 2):
        raise ValueError(
            "Rotation dataset requires sources, a reference, and rotated orientations"
        )
    expected_names = [
        f"Rotation-{label}-{orientation}"
        for label in metadata["sources"] for orientation in orientations
    ]
    selected_names = dataset["molecule_Rotation"]
    if (len(set(expected_names)) != len(expected_names)
            or len(set(selected_names)) != len(selected_names)
            or set(selected_names) != set(expected_names)
            or not set(expected_names).issubset(dataset["molecule"])):
        raise ValueError("Rotation metadata does not match the molecule lists")

    results = []
    for label in metadata["sources"]:
        reference_name = f"Rotation-{label}-reference"
        reference_atoms = dataset[reference_name]
        reference_coords = np.array(
            [atom[1:] for atom in reference_atoms], dtype=np.float64
        )
        for orientation, operations in orientations.items():
            name = f"Rotation-{label}-{orientation}"
            atoms = dataset[name]
            matrix = rotation_matrix(operations)
            coords = np.array([atom[1:] for atom in atoms], dtype=np.float64)
            if ([atom[0] for atom in atoms] != [atom[0] for atom in reference_atoms]
                    or coords.shape != reference_coords.shape
                    or not np.allclose(
                        coords, reference_coords @ matrix.T, rtol=0, atol=2e-12
                    )):
                raise ValueError(
                    f"{name}: geometry or atom order does not match the rotation"
                )
            for key in ("spin", "charge"):
                if dataset[key][name] != dataset[key][reference_name]:
                    raise ValueError(f"{name}: {key} differs from the reference")

            missing = list(dict.fromkeys(
                item for item in (reference_name, name) if item not in records
            ))
            row = dict(
                case=label, orientation=orientation, name=name,
                reference=reference_name,
                missing=missing, delta_hartree=None, delta_kcal=None, energy_pass=None,
                gradient_max_error=None, gradient_rms_error=None, gradient_pass=None,
                gradient_status="PENDING", missing_gradients=[],
            )
            if missing:
                if not completed_only:
                    raise ValueError(
                        f"Missing converged rotation results: {', '.join(missing)}"
                    )
                results.append(row)
                continue

            reference = records[reference_name]
            record = records[name]
            if reference["settings"] != record["settings"]:
                raise ValueError(f"{name}: inconsistent model/grid/precision settings")
            delta = float(record["energy"] - reference["energy"])
            if not np.isfinite(delta):
                raise ValueError(f"{name}: nonfinite energy difference")
            row.update(
                delta_hartree=delta, delta_kcal=delta * AU2KCALMOL,
                energy_pass=abs(delta) <= energy_tol,
            )

            reference_gradient = reference["gradient"]
            gradient = record["gradient"]
            if reference_gradient is None or gradient is None:
                row["missing_gradients"] = list(dict.fromkeys(
                    item for item, value in (
                        (reference_name, reference_gradient), (name, gradient)
                    )
                    if value is None
                ))
                row["gradient_status"] = (
                    "PENDING" if require_gradients or reference_gradient is not None
                    or gradient is not None else "NOT AVAILABLE"
                )
            else:
                reference_gradient = np.asarray(reference_gradient, dtype=np.float64)
                gradient = np.asarray(gradient, dtype=np.float64)
                if (reference_gradient.shape != reference_coords.shape
                        or gradient.shape != coords.shape):
                    raise ValueError(f"{name}: incorrect gradient shape")
                error = gradient - reference_gradient @ matrix.T
                if not np.isfinite(error).all():
                    raise ValueError(f"{name}: nonfinite gradient")
                maximum = float(np.max(np.abs(error)))
                rms = float(np.sqrt(np.mean(error ** 2)))
                passed = maximum <= gradient_tol
                row.update(
                    gradient_max_error=maximum, gradient_rms_error=rms,
                    gradient_pass=passed, gradient_status="PASS" if passed else "FAIL",
                )
            results.append(row)
    return results


def format_rotation_row(row):
    if row["missing"]:
        return (
            f"{row['name']}: PENDING "
            f"(missing/unconverged: {', '.join(row['missing'])})"
        )

    status = "PASS" if row["energy_pass"] else "FAIL"
    text = (f"{row['name']}: dE={row['delta_hartree']:+.8e} Eh "
            f"({row['delta_kcal']:+.6f} kcal/mol) [{status}]")
    if row["gradient_status"] in ("PASS", "FAIL"):
        text += (f"\n  gradient max={row['gradient_max_error']:.6e}, "
                 f"RMS={row['gradient_rms_error']:.6e} [{row['gradient_status']}]")
    elif row["gradient_status"] == "PENDING":
        text += (
            "\n  Gradient covariance: PENDING "
            f"(missing: {', '.join(row['missing_gradients'])})"
        )
    else:
        text += (
            "\n  Gradient covariance: NOT AVAILABLE "
            "(energy-only or missing gradients)."
        )
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Validation CSV file")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Rotation dataset JSON")
    parser.add_argument("--basis", required=True,
                        help="Basis suffix used in CSV record names")
    parser.add_argument("--energy-tol", type=float, default=1e-5,
                        help="Maximum absolute energy difference in Hartree")
    parser.add_argument("--gradient-tol", type=float, default=1e-5,
                        help="Maximum gradient component error in Hartree/Bohr")
    parser.add_argument("--completed-only", action="store_true",
                        help="Report available results and mark missing runs PENDING")
    parser.add_argument("--require-gradients", action="store_true",
                        help="Mark missing gradients PENDING, including energy-only")
    parser.add_argument("--grid-response", choices=("on", "off", "unknown"),
                        default="unknown",
                        help="Declared gradient setting; CSVs cannot verify it")
    args = parser.parse_args(argv)

    dataset = json.loads(args.dataset.read_text())
    records = read_force_csv(
        [args.csv], dataset, args.basis, completed_only=args.completed_only
    )
    results = rotation_report(dataset, records, args.energy_tol, args.gradient_tol,
                              completed_only=args.completed_only,
                              require_gradients=args.require_gradients)
    print("Energy differences: Hartree and kcal/mol; gradient errors: Hartree/Bohr")
    print(f"Energy tolerance: {args.energy_tol:.6e}; "
          f"gradient tolerance: {args.gradient_tol:.6e}; "
          f"declared grid response: {args.grid_response}")
    print("CSV does not verify model, grid, SCF tolerances, precision, rotation, "
          "geometry/atom order, units, or grid response.")
    for row in results:
        print(format_rotation_row(row))

    pending = sum(
        bool(row["missing"]) or row["gradient_status"] == "PENDING" for row in results
    )
    energy_failures = sum(row["energy_pass"] is False for row in results)
    gradient_failures = sum(row["gradient_pass"] is False for row in results)
    print(f"Orientations: {len(results)}; pending: {pending}; "
          f"energy failures: {energy_failures}; gradient failures: {gradient_failures}")
    if energy_failures or gradient_failures:
        return 1
    return 2 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
