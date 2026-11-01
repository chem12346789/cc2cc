"""GPU-resident Gaussian factors for full-grid occupied-virtual ZMP training."""

from __future__ import annotations

from collections import ChainMap
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cc2cc.utils.env_var import DATA_PATH, EDGE_LEN, EDGE_SIZE


class GaussianOrbitalCache:
    """Keep the complete fixed-reference grid factors resident on one device."""

    @classmethod
    def from_archive(
        cls, name, input_cpu, weight_cpu, device, block_size, input_tolerance=5e-8
    ):
        data_path = DATA_PATH / f"data_{name}.npz"
        addon_path = DATA_PATH / f"data_{name}_addon.npz"
        metadata_path = DATA_PATH / f"data_{name}_zmp_metadata.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"ZMP dataset not found: {data_path}")
        with ExitStack() as stack:
            archives = [
                stack.enter_context(np.load(data_path, allow_pickle=True))
            ]
            if addon_path.exists():
                archives.insert(
                    0,
                    stack.enter_context(np.load(addon_path, allow_pickle=True)),
                )
            if metadata_path.exists():
                archives.insert(
                    0,
                    stack.enter_context(np.load(metadata_path, allow_pickle=True)),
                )
            return cls(
                name,
                input_cpu,
                weight_cpu,
                ChainMap(*archives),
                device,
                block_size,
                input_tolerance,
            )

    def __init__(
        self,
        name,
        input_cpu,
        weight_cpu,
        data,
        device,
        block_size,
        input_tolerance=5e-8,
    ):
        self.name = name
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.block_size = block_size

        required = (
            "zmp_constraint_mol",
            "zmp_constraint_dm",
            "zmp_constraint_fock",
            "zmp_constraint_overlap",
            "zmp_constraint_mo_coeff",
            "zmp_constraint_mo_occ",
            "zmp_constraint_coords",
            "zmp_constraint_weights",
            "zmp_constraint_cube_size",
            "zmp_constraint_edge_len",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"{name}: missing ZMP orbital metadata {missing}; run "
                "gen_data.py --if_continue true --zmp_constraint_metadata true "
                "--zmp_feature_ov false --zmp_observable_response false."
            )

        input_cpu = input_cpu.detach().to(device="cpu")
        weight_cpu = weight_cpu.detach().to(device="cpu").reshape(-1)
        if input_cpu.dtype != torch.float64:
            raise ValueError("Full-OV ZMP training requires float64 model inputs.")
        if input_cpu.ndim != 3 or input_cpu.shape[-1] != EDGE_SIZE**3:
            raise ValueError(
                "Gaussian full-OV training requires cube inputs shaped "
                "(grid, feature, cube_sample)."
            )
        if int(np.asarray(data["zmp_constraint_cube_size"]).item()) != EDGE_SIZE**3:
            raise ValueError("Saved ZMP cube size differs from DFT2CC_EDGE_SIZE.")
        saved_edge_len = float(np.asarray(data["zmp_constraint_edge_len"]).item())
        if not np.isclose(saved_edge_len, EDGE_LEN, rtol=0, atol=1e-15):
            raise ValueError("Saved ZMP cube spacing differs from DFT2CC_EDGE_LEN.")

        self.ncenter, self.nfeature, self.ncube = input_cpu.shape
        self.coords_np = np.asarray(
            data["zmp_constraint_coords"], dtype=np.float64
        )
        self.weights_np = np.asarray(
            data["zmp_constraint_weights"], dtype=np.float64
        ).reshape(-1)
        if self.coords_np.shape != (self.ncenter, 3):
            raise ValueError(f"{name}: ZMP centers do not match model inputs.")
        if self.weights_np.shape != (self.ncenter,):
            raise ValueError(f"{name}: ZMP quadrature weights have an invalid shape.")
        if weight_cpu.shape != (self.ncenter,) or not np.allclose(
            weight_cpu.numpy(), self.weights_np, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(f"{name}: model and ZMP quadrature weights differ.")

        from pyscf import gto

        mol = gto.Mole().loads(str(np.asarray(data["zmp_constraint_mol"]).item()))
        dm = np.asarray(data["zmp_constraint_dm"], dtype=np.float64)
        coeff = np.asarray(data["zmp_constraint_mo_coeff"], dtype=np.float64)
        occupation = np.asarray(data["zmp_constraint_mo_occ"], dtype=np.float64)
        fock = np.asarray(data["zmp_constraint_fock"], dtype=np.float64)
        if dm.shape[0] != 1 or coeff.shape[0] != 1 or occupation.shape[0] != 1:
            raise ValueError("Gaussian full-OV training currently supports RKS only.")
        dm, coeff, occupation, fock = dm[0], coeff[0], occupation[0], fock[0]
        if mol.spin != 0 or coeff.shape != (mol.nao, mol.nao):
            raise ValueError(f"{name}: saved reference is not a closed-shell RKS system.")
        if input_cpu.shape[0] != len(self.coords_np):
            raise ValueError(f"{name}: model inputs and saved grid centers differ.")
        if len(occupation) != mol.nao or fock.shape != (mol.nao, mol.nao):
            raise ValueError(f"{name}: saved Fock/orbital dimensions are inconsistent.")
        if not np.all(np.isclose(occupation, 0) | np.isclose(occupation, 2)):
            raise ValueError("ZMP full-OV RKS training requires integer occupations 0/2.")
        overlap = np.asarray(data["zmp_constraint_overlap"], dtype=np.float64)
        if not np.allclose(
            (coeff * occupation) @ coeff.T, dm, rtol=0, atol=1e-7
        ):
            raise ValueError(f"{name}: saved orbitals do not reproduce the ZMP density.")
        if not np.allclose(
            coeff.T @ overlap @ coeff,
            np.eye(mol.nao),
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError(f"{name}: saved orbitals are not overlap-orthonormal.")
        occupied = occupation > 0
        occupied_mask = torch.as_tensor(
            occupied, dtype=torch.bool, device=self.device
        )
        self.coeff = torch.as_tensor(coeff, dtype=self.dtype, device=self.device)
        self.co = self.coeff[:, occupied_mask]
        self.cv = self.coeff[:, ~occupied_mask]
        self.fock0 = torch.as_tensor(fock, dtype=self.dtype, device=self.device)
        self.input = input_cpu.to(device=self.device, dtype=self.dtype)
        self.weights = torch.as_tensor(
            self.weights_np[:, None], dtype=self.dtype, device=self.device
        )

        self.coords = torch.as_tensor(
            self.coords_np, dtype=self.dtype, device=self.device
        )
        axis = (
            torch.arange(EDGE_SIZE, dtype=self.dtype, device=self.device)
            - EDGE_SIZE // 2
        ) * EDGE_LEN
        offsets = torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
        ).reshape(self.ncube, 3)
        self.sample_coords = self.coords[:, None, :] + offsets[None, :, :]
        self.atom_coords = torch.as_tensor(
            mol.atom_coords(), dtype=self.dtype, device=self.device
        )

        pair_indices = {}
        pair_atoms = []
        pair_exponents = []
        self.shells = []
        ao_loc = mol.ao_loc_nr()
        for shell_id in range(mol.nbas):
            atom_id = mol.bas_atom(shell_id)
            angular = mol.bas_angular(shell_id)
            exponents = np.asarray(mol.bas_exp(shell_id), dtype=np.float64)
            radial_ids = []
            for exponent in exponents:
                key = (atom_id, float(exponent))
                if key not in pair_indices:
                    pair_indices[key] = len(pair_atoms)
                    pair_atoms.append(atom_id)
                    pair_exponents.append(exponent)
                radial_ids.append(pair_indices[key])

            powers = [
                (lx, ly, angular - lx - ly)
                for lx in range(angular, -1, -1)
                for ly in range(angular - lx, -1, -1)
            ]
            self.shells.append(
                {
                    "atom": atom_id,
                    "radial_ids": torch.as_tensor(
                        radial_ids, dtype=torch.int64, device=self.device
                    ),
                    "exponents": torch.as_tensor(
                        exponents, dtype=self.dtype, device=self.device
                    ),
                    "coeff": torch.as_tensor(
                        np.ascontiguousarray(mol._libcint_ctr_coeff(shell_id)),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    "powers": torch.as_tensor(
                        powers, dtype=self.dtype, device=self.device
                    ),
                    "transform": torch.as_tensor(
                        np.ascontiguousarray(gto.cart2sph(angular, normalized=None)),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    "start": int(ao_loc[shell_id]),
                    "stop": int(ao_loc[shell_id + 1]),
                    "angular": angular,
                }
            )

        self.pair_atoms = torch.as_tensor(
            pair_atoms, dtype=torch.int64, device=self.device
        )
        self.pair_exponents = torch.as_tensor(
            pair_exponents, dtype=self.dtype, device=self.device
        )
        self.radial = torch.empty(
            (self.ncenter, self.ncube, len(pair_atoms)),
            dtype=self.dtype,
            device=self.device,
        )
        self.chain = torch.empty(
            (self.ncenter, self.nfeature, self.ncube, 4),
            dtype=self.dtype,
            device=self.device,
        )

        if self.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            tile_bytes = (
                block_size
                * self.ncube
                * (
                    len(pair_atoms) * 3
                    + 4 * mol.nao
                    + self.nfeature * 4
                )
                * torch.tensor([], dtype=self.dtype).element_size()
            )
            if free_bytes < tile_bytes:
                raise MemoryError(
                    f"{name}: one configured GPU tile needs at least "
                    f"{tile_bytes / 2**30:.2f} GiB of additional workspace; "
                    f"only {free_bytes / 2**30:.2f} GiB is currently free."
                )

        self._build_radial_cache()
        self._build_chain_cache(mol, dm, input_cpu, input_tolerance)
        del self.coords_np, self.weights_np

    def _build_radial_cache(self):
        atom_coords = self.atom_coords.index_select(0, self.pair_atoms)
        for start in range(0, self.ncenter, self.block_size):
            stop = min(start + self.block_size, self.ncenter)
            delta = (
                self.sample_coords[start:stop, :, None, :]
                - atom_coords[None, None, :, :]
            )
            distance_squared = torch.sum(delta * delta, dim=-1)
            self.radial[start:stop] = torch.exp(
                -distance_squared * self.pair_exponents[None, None, :]
            )

    def _build_chain_cache(self, mol, dm, input_cpu, input_tolerance):
        from pyscf.dft import numint

        from cc2cc.utils.Grids import GridCube

        cube_axis = (
            np.arange(EDGE_SIZE, dtype=np.float64) - EDGE_SIZE // 2
        ) * EDGE_LEN
        offsets = np.stack(
            np.meshgrid(cube_axis, cube_axis, cube_axis, indexing="ij"), axis=-1
        )
        grid = SimpleNamespace(
            mol=mol,
            input_level=self.nfeature,
            cube_type="cube",
            cube_size=self.ncube,
            cutoff=1e-15,
        )
        ni = numint.NumInt()
        for start in range(0, self.ncenter, self.block_size):
            stop = min(start + self.block_size, self.ncenter)
            cube_coords = (
                self.coords_np[start:stop, None, None, None, :]
                + offsets[None, :, :, :, :]
            )
            cube = GridCube(cube_coords, grid)
            features, chain, _ = cube.gen_cube_rho_rks(ni, dm, ao_deriv=1)
            reference = input_cpu[start:stop].numpy()
            if not np.allclose(
                features, reference, rtol=input_tolerance, atol=1e-10
            ):
                difference = float(np.max(np.abs(features - reference)))
                raise ValueError(
                    f"{self.name}: regenerated PySCF features differ from the "
                    f"training inputs (max abs error {difference:.3e})."
                )
            chain = np.ascontiguousarray(chain.transpose(2, 0, 3, 1))
            self.chain[start:stop] = torch.as_tensor(
                chain, dtype=self.dtype, device=self.device
            )

    def fock_tile(self, feature_gradient, start):
        stop = start + len(feature_gradient)
        batch = len(feature_gradient)
        ao = torch.zeros(
            (batch, self.ncube, 4, self.coeff.shape[0]),
            dtype=self.dtype,
            device=self.device,
        )
        sample_coords = self.sample_coords[start:stop]

        for shell in self.shells:
            delta = sample_coords - self.atom_coords[shell["atom"]]
            radial = self.radial[start:stop].index_select(
                -1, shell["radial_ids"]
            )
            contracted = radial @ shell["coeff"]
            powers = shell["powers"]
            cartesian = torch.prod(
                delta.unsqueeze(-2).pow(powers), dim=-1
            )
            angular = cartesian @ shell["transform"]
            shell_width = 2 * shell["angular"] + 1
            for contraction in range(shell["coeff"].shape[1]):
                ao_start = shell["start"] + contraction * shell_width
                ao_stop = ao_start + shell_width
                values = contracted[:, :, contraction, None] * angular
                ao[:, :, 0, ao_start:ao_stop] = values
                for axis in range(3):
                    reduced_powers = powers.clone()
                    derivative_order = reduced_powers[:, axis].clone()
                    reduced_powers[:, axis] = torch.clamp(
                        reduced_powers[:, axis] - 1, min=0
                    )
                    dcartesian = (
                        torch.prod(
                            delta.unsqueeze(-2).pow(reduced_powers), dim=-1
                        )
                        * derivative_order
                    )
                    dangular = dcartesian @ shell["transform"]
                    dradial = (
                        radial
                        * (
                            -2
                            * delta[:, :, axis, None]
                            * shell["exponents"][None, None, :]
                        )
                    ) @ shell["coeff"]
                    derivative = (
                        dradial[:, :, contraction, None] * angular
                        + contracted[:, :, contraction, None] * dangular
                    )
                    ao[:, :, axis + 1, ao_start:ao_stop] = derivative

        chain = self.chain[start:stop]
        weighted_density = torch.einsum(
            "blq,blqd->bqd", feature_gradient, chain
        )
        phi = ao[:, :, 0, :]
        fock = torch.einsum(
            "bq,bqi,bqj->ij", weighted_density[:, :, 0], phi, phi
        )
        for axis in range(1, 4):
            derivative = ao[:, :, axis, :]
            fock = fock + torch.einsum(
                "bq,bqi,bqj->ij",
                weighted_density[:, :, axis],
                derivative,
                phi,
            )
            fock = fock + torch.einsum(
                "bq,bqi,bqj->ij",
                weighted_density[:, :, axis],
                phi,
                derivative,
            )
        return fock

    def batches(self):
        for start in range(0, self.ncenter, self.block_size):
            yield start, min(start + self.block_size, self.ncenter)
