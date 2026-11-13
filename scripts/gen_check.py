"""Recompute saved grid-energy errors: python -m scripts.gen_check.

Read train/eval molecules from configs/gen_mol2.json by default. Print errors
whose magnitude exceeds --threshold times the atom count in kcal/mol,
preserving their signs.
Warn and continue when data files are missing or malformed.
Finish with counts of checked, missing, invalid, and above-threshold results.
The current NPZ files are used, so saved reruns replace earlier results.
Use --train or --eval to check only that split; the default checks both.
Moving-grid force-training data is always checked for every selected archive.
Force residuals use --force-threshold, separately from kcal/mol energy errors.
"""

import argparse
from pathlib import Path
import sys
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc2cc.utils.helper.runtime.env_var import DATA_PATH
from cc2cc.utils.helper.units import AU2KCALMOL
from cc2cc.utils.parser import config_list, gen_name_args, process_config


def calculate_errors(path):
    """Return the four generation diagnostics in kcal/mol from one saved NPZ."""
    energy_keys = ("e_cc", "e_dft", "e_hf", "e_zmp")
    grid_keys = (
        "weights", "tol_delta_grids", "tol_delta_zmp_grids",
        "exc_cc_grids", "hatree_cc_grids", "kin_cc_grids", "kinl_cc_grids",
        "nuc_cc_grids", "ecp_cc_grids",
        "exc_k_hf_grids", "hatree_hf_grids", "kin_hf_grids",
        "nuc_hf_grids", "ecp_hf_grids",
        "exc_dft_grids", "exc_k_dft_grids", "hatree_dft_grids",
        "kinl_dft_grids", "nuc_dft_grids", "ecp_dft_grids",
        "grid_weight_response", "grid_weight_response_zmp",
    )
    response_keys = ("grid_weight_response", "grid_weight_response_zmp")
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"{path}: expected an NPZ archive")
    with archive:
        missing = sorted(set(energy_keys + grid_keys) - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
        data = {key: archive[key] for key in energy_keys + grid_keys}
        weights = data["weights"]
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError(f"{path}: weights must be a nonempty 1-D array")
        for key, value in data.items():
            if key in response_keys:
                gradient = archive["grad_cc"]
                if gradient.ndim != 2 or gradient.shape[1] != 3 or not gradient.shape[0]:
                    raise ValueError(f"{path}: grad_cc must have shape (natm, 3)")
                expected_shape = (len(gradient), 3, len(weights))
            else:
                expected_shape = () if key in energy_keys else weights.shape
            if value.shape != expected_shape:
                raise ValueError(
                    f"{path}: {key} has shape {value.shape}; expected {expected_shape}"
                )
            if value.dtype.kind not in "fiu":
                raise ValueError(f"{path}: {key} must contain real numeric values")
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: {key} contains non-finite values")
        weights = data["weights"]
        e_cc = data["e_cc"]
        e_dft = data["e_dft"]
        e_hf = data["e_hf"]
        e_zmp = data["e_zmp"]
        cc_kin = (
            data["exc_cc_grids"]
            + data["hatree_cc_grids"]
            + data["kin_cc_grids"]
            + data["nuc_cc_grids"]
            + data["ecp_cc_grids"]
        )
        hf_kin = (
            data["exc_k_hf_grids"]
            + data["hatree_hf_grids"]
            + data["kin_hf_grids"]
            + data["nuc_hf_grids"]
            + data["ecp_hf_grids"]
        )
        cc_kinl = (
            data["exc_cc_grids"]
            + data["hatree_cc_grids"]
            + data["kinl_cc_grids"]
            + data["nuc_cc_grids"]
            + data["ecp_cc_grids"]
        )
        dft_kinl = (
            data["exc_dft_grids"]
            + data["exc_k_dft_grids"]
            + data["hatree_dft_grids"]
            + data["kinl_dft_grids"]
            + data["nuc_dft_grids"]
            + data["ecp_dft_grids"]
        )
        # Saved energy_train uses the ZMP baseline, not the DFT baseline.
        errors = {
            "Error": np.sum(data["tol_delta_grids"] * weights) - (e_cc - e_dft),
            "Error HF part": np.sum((cc_kin - hf_kin) * weights) - (e_cc - e_hf),
            "Error exc_kinl part": (
                np.sum((cc_kinl - dft_kinl) * weights) - (e_cc - e_dft)
            ),
            "Error ZMP": (
                np.sum(data["tol_delta_zmp_grids"] * weights) - (e_cc - e_zmp)
            ),
        }
    results = {}
    for label, error in errors.items():
        value = float(error * AU2KCALMOL)
        if not np.isfinite(value):
            raise ValueError(f"{path}: {label} is non-finite: {value}")
        results[label] = value
    return results


def calculate_force_errors(path):
    """Validate force arrays and return residuals with the atom count."""
    with np.load(path, allow_pickle=False) as data:
        def read_array(key, shape):
            if key not in data:
                raise ValueError(f"{path}: missing {key}; regenerate force mappings")
            value = data[key]
            if value.shape != shape:
                raise ValueError(
                    f"{path}: {key} has shape {value.shape}; expected {shape}"
                )
            if value.dtype.kind != "f":
                raise ValueError(f"{path}: {key} must contain real floating-point values")
            for block in value:
                if not np.isfinite(block).all():
                    raise ValueError(f"{path}: {key} contains non-finite values")
            return value

        weights = data["weights"]
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError(f"{path}: weights must be a nonempty 1-D array")
        ngrid = len(weights)
        grad_cc = data["grad_cc"]
        if grad_cc.ndim != 2 or grad_cc.shape[1] != 3 or not grad_cc.shape[0]:
            raise ValueError(f"{path}: grad_cc must have shape (natm, 3)")
        natm = len(grad_cc)
        gradients = {
            key: read_array(key, (natm, 3))
            for key in ("grad_cc", "grad_dft", "grad_dft_d3bj", "grad_zmp", "grad_cc_train")
        }
        errors = {
            "Gradient target (Hartree/Bohr)": float(np.max(np.abs(
                gradients["grad_cc_train"] - (gradients["grad_cc"] - gradients["grad_dft"])
            ))),
        }
        for suffix in ("dft", "zmp"):
            rho_key = f"rho_cube_{suffix}"
            rho = data[rho_key]
            if (
                rho.ndim != 5 or rho.shape[0] != ngrid or rho.shape[1] == 0
                or rho.shape[2] == 0 or len(set(rho.shape[2:])) != 1
            ):
                raise ValueError(f"{path}: {rho_key} must have shape (ngrids, ninput, edge, edge, edge)")
            rho_shape = rho.shape
            del rho
            mapping_key = "grad2force" if suffix == "dft" else "grad2force_zmp"
            mapping = read_array(
                mapping_key, (natm, rho_shape[1], ngrid, *rho_shape[2:], 3)
            )
            translation_error = 0.0
            # Bound temporary reductions for the large feature Jacobian.
            for start in range(0, ngrid, 128):
                stop = start + 128
                residual = mapping[:, :, start:stop].sum(axis=0)
                residual = residual[:, weights[start:stop] != 0]
                translation_error = max(
                    translation_error, float(np.max(np.abs(residual), initial=0))
                )
            del mapping
            errors[f"{suffix} feature translation (feature/Bohr)"] = translation_error
            response_key = (
                "grid_weight_response" if suffix == "dft" else "grid_weight_response_zmp"
            )
            response = read_array(response_key, (natm, 3, ngrid))
            errors[f"{suffix} weight translation (weight/Bohr)"] = float(
                np.max(np.abs(response.sum(axis=0)))
            )
            del response
    return errors, natm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen_config", default="gen_mol2.json")
    parser.add_argument("--dataset", default="gmtkn-def2")
    parser.add_argument("--basis", default="def2-QZVPPD")
    parser.add_argument("--data_dir", type=Path, default=DATA_PATH)
    split_args = parser.add_mutually_exclusive_group()
    split_args.add_argument(
        "--split", choices=("train", "eval", "both"), default="both"
    )
    split_args.add_argument(
        "--train", dest="split", action="store_const", const="train",
        help="Check only training molecules.",
    )
    split_args.add_argument(
        "--eval", dest="split", action="store_const", const="eval",
        help="Check only evaluation molecules.",
    )
    parser.add_argument(
        "--threshold", type=float, default=5e-3,
        help="Per-atom threshold; print errors above this value times atom count in kcal/mol.",
    )
    parser.add_argument(
        "--force-threshold", type=float, default=1e-6,
        help="Absolute residual threshold in each force diagnostic's displayed units.",
    )
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and nonnegative")
    if not np.isfinite(args.force_threshold) or args.force_threshold < 0:
        parser.error("--force-threshold must be finite and nonnegative")
    config = process_config(args.gen_config)
    splits = ("train", "eval") if args.split == "both" else (args.split,)
    configured_names = [
        name for split in splits for name in config_list(config, split)
    ]
    names = gen_name_args(configured_names, args.dataset)
    checked = missing = invalid = outlier_molecules = 0
    outliers_by_label = {}
    force_outliers_by_label = {}
    for name in names:
        path = args.data_dir / f"data_{name}_{args.basis}.npz"
        if not path.is_file():
            missing += 1
            print(
                f"Warning: {name}: missing generated data file: {path}",
                file=sys.stderr,
                flush=True,
            )
            continue
        try:
            errors = calculate_errors(path)
            force_errors, natm = calculate_force_errors(path)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile, ZlibError) as error:
            invalid += 1
            print(
                f"Warning: {name}: invalid generated data file: {path}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        checked += 1
        has_outlier = False
        threshold = args.threshold * natm
        for label, error in errors.items():
            outliers_by_label.setdefault(label, 0)
            if abs(error) > threshold:
                has_outlier = True
                outliers_by_label[label] += 1
                print(f"{name}: {label} (kcal/mol): {error}", flush=True)
        for label, error in force_errors.items():
            force_outliers_by_label.setdefault(label, 0)
            if abs(error) > args.force_threshold:
                has_outlier = True
                force_outliers_by_label[label] += 1
                print(f"{name}: {label}: {error}", flush=True)
        outlier_molecules += int(has_outlier)
    print(
        f"Summary ({args.split}): requested={len(names)}, checked={checked}, "
        f"missing={missing}, invalid={invalid}, "
        f"outlier_molecules={outlier_molecules}, "
        f"outlier_diagnostics={sum(outliers_by_label.values()) + sum(force_outliers_by_label.values())}",
        flush=True,
    )
    for label, count in outliers_by_label.items():
        print(
            f"  {label}: {count} above {args.threshold:g} kcal/mol per atom "
            "in magnitude"
        )
    for label, count in force_outliers_by_label.items():
        print(f"  {label}: {count} above {args.force_threshold:g} in magnitude")


if __name__ == "__main__":
    main()
