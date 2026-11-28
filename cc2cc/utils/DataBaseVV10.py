"""Load four spatial features and a density-matched, center-only VV10 channel."""

from collections import ChainMap

import numpy as np
import torch
from pyscf import gto

from cc2cc.utils.DataBase import DataBase, MAX_GRAD_BATCH_SIZE
from cc2cc.utils.helper.chemistry.vv10_input import vv10_input_data
from cc2cc.utils.helper.runtime.env_var import CUBE_MIDDLE


class DataBaseVV10(DataBase):
    """Reuse the standard targets and batching with VV10 as input channel five."""

    def _load_data(self, mol_info, name, data):
        if self.dtype != torch.float64:
            raise ValueError("VV10-input models require --precision float64.")
        rho_input = self.args.rho_input
        if rho_input == "dft_vv10" or (rho_input == "zmp_vv10" and self.if_eval):
            rho_key = "rho_cube_scf_vv10"
            vv10_key = "exc_scf_vv10_grid"
            dm_key = "dm1_scf_vv10"
            grad_key = "grad2force_scf_vv10"
            weight_key = "grid_weight_response_scf_vv10"
        elif rho_input in ("zmp", "zmp_vv10") and not self.if_eval:
            rho_key = "rho_cube_zmp"
            vv10_key = "exc_post_vv10_grid_zmp"
            dm_key = "dm1_zmp"
            grad_key = "grad2force_zmp"
            weight_key = "grid_weight_response_zmp"
        elif rho_input in ("dft", "dft_d3bj", "dft_d3bj_1", "zmp"):
            rho_key = "rho_cube_dft"
            vv10_key = (
                "exc_post_vv10_grid_pbe"
                if self.baseline == "pbe"
                else "exc_post_vv10_grid"
            )
            dm_key = "dm1_dft"
            grad_key = "grad2force"
            weight_key = "grid_weight_response"
        else:
            raise ValueError(f"No VV10 input mapping for --rho_input {rho_input}.")

        pbe_data = (
            data.maps[0]
            if self.baseline == "pbe" and isinstance(data, ChainMap)
            else data
        )
        density_data = (
            pbe_data if self.baseline == "pbe" and rho_key == "rho_cube_dft" else data
        )
        vv10_data = density_data if vv10_key == "exc_post_vv10_grid_pbe" else data
        overrides = {}
        mol = None
        dm = None
        unrestricted = None
        force_response = None
        weight_response = None
        if rho_key in density_data:
            rho = density_data[rho_key]
        else:
            from pyscf.dft.numint import NumInt
            from cc2cc.utils.reference.reference_b3lyp import get_dft_input

            if dm_key not in density_data:
                raise ValueError(f"{name}: missing density matrix {dm_key}.")
            mol = self._molecule(mol_info, name, data)
            dm = np.asarray(density_data[dm_key], dtype=np.float64)
            unrestricted = bool(np.asarray(data["unrestricted"]).item())
            grid = self._rebuild_grid(mol, name, data)
            self.print(f"{name}: reconstructing missing input features.")
            rho = get_dft_input(
                mol, grid, dm, unrestricted=unrestricted, ni=NumInt()
            )["rho_cube_dft"]

        input_level = 6 if self.baseline == "pbe" else 5
        features = np.zeros(
            (rho.shape[0], input_level, *rho.shape[2:]), dtype=np.float64
        )
        features[:, :4] = rho[:, :4]
        if input_level == 6 and rho.shape[1] > 5:
            features[:, 5] = rho[:, 5]

        need_force = (
            self.args.if_grad and not self.if_eval and len(rho) < MAX_GRAD_BATCH_SIZE
        )
        need_response = self.if_zmp_response and not self.if_eval
        if self.baseline == "pbe":
            self._pbe_targets(data, pbe_data, name, need_force, need_response, overrides)
        elif not self.if_eval:
            if (
                need_force and rho_input == "dft" and "grad_cc_train" not in data
                and "grad_cc" in data and "grad_dft" in data
            ):
                overrides["grad_cc_train"] = data["grad_cc"] - data["grad_dft"]
            if self.args.if_abs and rho_input != "dft_vv10":
                zmp = rho_input in ("zmp", "zmp_vv10")
                target_key = "tol_delta_zmp_grids" if zmp else "tol_delta_grids"
                baseline_key = "tol_zmp_grids" if zmp else "tol_dft_grids"
                if (
                    target_key not in data and "tol_cc_grids" in data
                    and baseline_key in data
                ):
                    overrides[target_key] = data["tol_cc_grids"] - data[baseline_key]
        response_data = None
        projections = ()
        if need_response:
            response_data = super()._load_zmp_response_data(
                data, name, self.process_input(features)[:, :4]
            )
            projections = self._projection_densities(data, name)

        if vv10_key not in vv10_data or need_force or need_response:
            if dm_key not in density_data:
                raise ValueError(f"{name}: missing density matrix {dm_key}.")
            if mol is None:
                mol = self._molecule(mol_info, name, data)
                dm = np.asarray(density_data[dm_key], dtype=np.float64)
                unrestricted = bool(np.asarray(data["unrestricted"]).item())
            owners = None
            if need_force:
                owners, weight_response = self._force_mappings(
                    mol, name, data, dm, unrestricted, grad_key, weight_key,
                    overrides, grad_key in density_data,
                )
            self.print(f"{name}: evaluating VV10 input and requested derivatives.")
            vv10, force_response, projected_response = vv10_input_data(
                mol,
                data["coords"],
                data["weights"],
                dm,
                unrestricted=unrestricted,
                weight_response=weight_response,
                grid_owners=owners,
                projection_dms=projections,
            )
            if vv10_key in vv10_data:
                cached = np.asarray(vv10_data[vv10_key], dtype=np.float64).reshape(-1)
                if cached.shape != vv10.shape or not np.allclose(
                    cached, vv10, rtol=5e-7, atol=1e-10
                ):
                    raise ValueError(
                        f"{name}: {vv10_key} does not match the saved {dm_key} "
                        "and integration grid."
                    )
                vv10 = cached
            if response_data is not None:
                for spin, response in enumerate(projected_response):
                    response_data[f"zmp_response_vv10_features_s{spin}"] = response
        else:
            vv10 = np.asarray(vv10_data[vv10_key], dtype=np.float64).reshape(-1)
        if vv10.size != rho.shape[0]:
            raise ValueError(f"{name}: {vv10_key} and {rho_key} have different grid sizes.")

        features[:, 4, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] = vv10
        overrides[rho_key] = features
        overrides[vv10_key] = vv10
        if response_data is not None:
            overrides["_vv10_response_data"] = response_data
        if rho_input == "zmp_vv10" and not self.if_eval:
            if "e_post_vv10_zmp" not in data:
                overrides["e_post_vv10_zmp"] = data["e_zmp"] + np.dot(
                    data["weights"], vv10
                )

        if need_force:
            if force_response is None or weight_response is None:
                raise RuntimeError(f"{name}: VV10 nuclear response was not produced.")
            if rho_input == "zmp_vv10" and "grad_post_vv10_zmp" not in data:
                correction = (
                    np.asarray(data["weights"]) @ force_response
                    + vv10 @ weight_response.transpose(2, 0, 1).reshape(len(vv10), -1)
                )
                overrides["grad_post_vv10_zmp"] = (
                    data["grad_zmp"] + correction.reshape(mol.natm, 3)
                )
            mapping = (
                overrides[grad_key] if grad_key in overrides else density_data[grad_key]
            )
            if mapping.shape[1] < input_level:
                expanded = np.zeros(
                    (mapping.shape[0], input_level, *mapping.shape[2:]),
                    dtype=np.float64,
                )
                expanded[:, :mapping.shape[1]] = mapping
                overrides[grad_key] = expanded

        count, batch = super()._load_data(mol_info, name, ChainMap(overrides, data))
        if force_response is not None:
            batch["grad2force"][:, 4] = 0
            middle = (
                CUBE_MIDDLE * rho.shape[2] * rho.shape[3]
                + CUBE_MIDDLE * rho.shape[3]
                + CUBE_MIDDLE
            )
            batch["grad2force"][:, 4, middle] = torch.as_tensor(
                force_response, dtype=torch.float64, device="cpu"
            )
        return count, batch

    def _load_zmp_response_data(self, data, name, processed_input):
        return data["_vv10_response_data"]

    def _pbe_targets(self, data, pbe, name, need_force, need_response, overrides):
        zmp = self.args.rho_input == "zmp" and not self.if_eval
        required = ["e_zmp" if zmp else "e_dft"]
        if need_force:
            if zmp:
                required.append("grad_zmp")
            elif "grad_cc_train" not in pbe:
                if "grad_dft" in pbe and "grad_cc" in data:
                    overrides["grad_cc_train"] = data["grad_cc"] - pbe["grad_dft"]
                else:
                    required.append("grad_cc_train")
        if self.args.if_abs and not self.if_eval:
            target_key = "tol_delta_zmp_grids" if zmp else "tol_delta_grids"
            baseline_key = "tol_zmp_grids" if zmp else "tol_dft_grids"
            if target_key not in pbe:
                if baseline_key in pbe and "tol_cc_grids" in data:
                    overrides[target_key] = data["tol_cc_grids"] - pbe[baseline_key]
                else:
                    required.append(target_key)
        if need_response:
            required.extend((
                "zmp_constraint_baseline_xc",
                "zmp_constraint_mo_coeff",
                "zmp_constraint_mo_occ",
                "zmp_response_projection_rcond",
                "zmp_observable_reference",
                "zmp_observable_target",
            ))
            nspin = 2 if bool(np.asarray(data["unrestricted"]).item()) else 1
            for spin in range(nspin):
                required.extend(
                    f"{prefix}_s{spin}"
                    for prefix in (
                        "zmp_potential_feature_file",
                        "zmp_potential_ov",
                        "zmp_observable_response",
                        "zmp_baseline_potential_coeff",
                    )
                )
        missing = [key for key in required if key not in pbe]
        if missing:
            raise ValueError(f"{name}: PBE supplement is missing {missing}.")

    def _molecule(self, mol_info, name, data):
        for key in ("pbe_mol", "zmp_constraint_mol"):
            if key in data:
                mol = gto.Mole().loads(str(np.asarray(data[key]).item()))
                break
        else:
            mol = mol_info["mol"]
            xyz = str(np.asarray(data["mol"]).item())
            if mol.tostring(format="xyz") != xyz:
                atoms = [
                    (parts[0], tuple(float(value) for value in parts[1:4]))
                    for parts in (line.split() for line in xyz.splitlines()[2:])
                ]
                mol = mol.set_geom_(atoms, unit="Angstrom", inplace=False)
                mol = self._restore_geometry(mol, name, data)
        if mol.tostring(format="xyz") != str(np.asarray(data["mol"]).item()):
            raise ValueError(f"{name}: saved molecule metadata does not match the archive.")
        return mol

    def _restore_geometry(self, mol, name, data):
        from cc2cc.utils.Grids import Grid

        level = int(data.get("zmp_constraint_grid_level", self.args.grid_level))
        grid = Grid(mol, level, 7)
        if grid.coords is None or grid.atm_idx is None:
            raise ValueError(f"{name}: rebuilt grid has no atom-centered coordinates.")
        coords = np.asarray(data["coords"], dtype=np.float64)
        grid_coords = np.asarray(grid.coords, dtype=np.float64)
        grid_owners = np.asarray(grid.atm_idx)
        owners = np.asarray(data.get("grid_atom_indices", grid_owners))
        if (
            coords.shape != grid_coords.shape
            or owners.shape != grid_owners.shape
            or not np.array_equal(owners, grid_owners)
        ):
            raise ValueError(f"{name}: cannot recover the saved molecular geometry.")
        geometry = mol.atom_coords()
        for atom in range(mol.natm):
            moved = owners == atom
            if not np.any(moved):
                raise ValueError(f"{name}: missing atom-centered grid for atom {atom}.")
            displacement = coords[moved] - grid_coords[moved]
            shift = displacement.mean(axis=0)
            if not np.allclose(displacement, shift, rtol=0, atol=1e-10):
                raise ValueError(f"{name}: cannot recover the saved molecular geometry.")
            geometry[atom] += shift
        return mol.set_geom_(geometry, unit="Bohr", inplace=False)

    def _rebuild_grid(self, mol, name, data):
        from cc2cc.utils.Grids import Grid

        level = int(data.get("zmp_constraint_grid_level", self.args.grid_level))
        grid = Grid(mol, level, 7)
        for key in ("coords", "weights"):
            saved = np.asarray(data[key])
            rebuilt = getattr(grid, key)
            if saved.shape != rebuilt.shape or not np.allclose(
                saved, rebuilt, rtol=0, atol=1e-10
            ):
                raise ValueError(
                    f"{name}: cannot reconstruct missing grid data; "
                    f"the rebuilt grid has different {key}."
                )
        return grid

    def _force_mappings(
        self, mol, name, data, dm, unrestricted, grad_key, weight_key, overrides,
        has_grad_mapping,
    ):
        need_grid = (
            not has_grad_mapping
            or weight_key not in data
            or "grid_atom_indices" not in data
        )
        owners = np.asarray(data["grid_atom_indices"]) if "grid_atom_indices" in data else None
        if need_grid:
            grid = self._rebuild_grid(mol, name, data)
            if owners is not None and (
                grid.atm_idx is None or not np.array_equal(owners, grid.atm_idx)
            ):
                raise ValueError(f"{name}: saved grid atom indices are misaligned.")
            if not has_grad_mapping:
                from cc2cc.utils.reference.reference_b3lyp import get_dft_grad

                self.print(f"{name}: reconstructing missing local force mappings.")
                mappings = get_dft_grad(mol, grid, dm, unrestricted=unrestricted)
                overrides[grad_key] = mappings["grad2force"]
                overrides[weight_key] = mappings["grid_weight_response"]

            if owners is None or (
                weight_key not in data and weight_key not in overrides
            ):
                from pyscf.grad.rks import grids_response_cc

                grid_owners = np.full(len(data["weights"]), -1, dtype=np.int32)
                weight_response = np.zeros(
                    (mol.natm, 3, len(grid_owners)), dtype=np.float64
                )
                offset = 0
                for atom, (coords, weights, response) in enumerate(grids_response_cc(grid)):
                    stop = offset + len(weights)
                    if not (
                        np.allclose(data["coords"][offset:stop], coords, rtol=0, atol=1e-12)
                        and np.allclose(
                            data["weights"][offset:stop], weights, rtol=1e-12, atol=1e-15
                        )
                    ):
                        raise ValueError(f"{name}: reconstructed grid response is misaligned.")
                    grid_owners[offset:stop] = atom
                    weight_response[:, :, offset:stop] = response
                    offset = stop
                if np.any(np.asarray(data["weights"])[offset:] != 0):
                    raise ValueError(f"{name}: nonzero weights follow the atom-centered grid.")
                if owners is None:
                    owners = grid_owners
                if weight_key not in data and weight_key not in overrides:
                    overrides[weight_key] = weight_response
        weight_response = (
            overrides[weight_key] if weight_key in overrides else data[weight_key]
        )
        return owners, np.asarray(weight_response, dtype=np.float64)

    @staticmethod
    def _projection_densities(data, name):
        required = (
            "zmp_constraint_mo_coeff",
            "zmp_constraint_mo_occ",
            "zmp_response_projection_rcond",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{name}: missing ZMP response metadata {missing}.")
        coefficients = np.asarray(data["zmp_constraint_mo_coeff"], dtype=np.float64)
        occupations = np.asarray(data["zmp_constraint_mo_occ"], dtype=np.float64)
        nspin = 2 if bool(np.asarray(data["unrestricted"]).item()) else 1
        if len(coefficients) != nspin or len(occupations) != nspin:
            raise ValueError(f"{name}: ZMP orbital metadata has inconsistent spin axes.")
        projected = []
        for spin, (coeff, occ) in enumerate(zip(coefficients, occupations)):
            key = f"zmp_potential_ov_s{spin}"
            if key not in data:
                raise ValueError(
                    f"{name}: missing {key}; regenerate ZMP observable response data."
                )
            potential = np.asarray(data[key], dtype=np.float64)
            co, cv = coeff[:, occ > 0], coeff[:, occ == 0]
            if potential.ndim != 3 or potential.shape[1:] != (cv.shape[1], co.shape[1]):
                raise ValueError(f"{name}: {key} has inconsistent orbital dimensions.")
            projection = np.linalg.pinv(
                potential.reshape(len(potential), -1).T,
                rcond=float(data["zmp_response_projection_rcond"]),
            ).reshape(len(potential), cv.shape[1], co.shape[1])
            dm = np.einsum("pa,kai,qi->kpq", cv, projection, co, optimize=True)
            # Match the existing sidecar's orbital-Jacobian normalization.
            projected.append((dm + dm.transpose(0, 2, 1)) * 0.5)
        return projected
