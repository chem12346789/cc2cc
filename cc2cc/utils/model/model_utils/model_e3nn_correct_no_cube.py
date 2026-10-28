"""Distance-aware E3nn with per-input edge vectors in Bohr."""

from typing import cast
import math

import torch
from e3nn import o3

class E3nn(torch.nn.Module):
    """The model_e3nn_correct architecture with dynamic neighborhood geometry."""

    def __init__(
        self, cube_type: str, cube_size: int, input_level: int, lmax: int,
        distance: float = 0.5,
    ):
        super().__init__()
        if cube_type != "neighbors":
            raise NotImplementedError("Only neighbor inputs are implemented.")
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("distance must be positive and finite, in Bohr.")
        self.cube_type = cube_type
        self.cube_size = cube_size
        self.input_level = input_level
        self.lmax = lmax
        self.distance = distance
        irreps_input = o3.Irreps(f"{input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join(f"{input_level}x{i}e" for i in range(lmax + 1))
        )
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_input, self.irreps_sh, hidden_irreps,
            shared_weights=False, internal_weights=False,
        ).to(device="cpu", dtype=torch.float64)
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(1, 16, device="cpu", dtype=torch.float64),
            torch.nn.GELU(),
            torch.nn.Linear(
                16, cast(int, self.tp1.weight_numel), device="cpu", dtype=torch.float64
            ),
        )
        self.readout = o3.Linear(
            hidden_irreps, o3.Irreps(f"{cube_size}x0e")
        ).to(device="cpu", dtype=torch.float64)

    def forward(self, f_in: torch.Tensor, edge_vec: torch.Tensor) -> torch.Tensor:
        """Map [..., cube_size, input_level] and [..., cube_size, 3] to scalars."""
        if f_in.shape[-2:] != (self.cube_size, self.input_level):
            raise ValueError("Feature shape does not match neighborhood size/channels.")
        if edge_vec.shape[-2:] != (self.cube_size, 3):
            raise ValueError("Expected edge_vec with shape [..., cube_size, 3].")
        radii = torch.linalg.vector_norm(edge_vec, dim=-1, keepdim=True) / self.distance
        sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        f_hidden = self.tp1(f_in, sh, self.radial(radii))
        return self.readout(f_hidden.mean(dim=-2, keepdim=True))
