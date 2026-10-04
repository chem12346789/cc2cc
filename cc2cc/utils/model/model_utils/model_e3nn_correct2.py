"""Distance-aware replacement for model_e3nn_correct.E3nn.

Import E3nn from this module to opt in; existing models are unchanged.
The constructor and input/output shapes are unchanged, but checkpoints are not
interchangeable. Distances are measured from each grid point to the cube center,
in the same units as EDGE_LEN, not between all pairs of grid points.
"""

from typing import cast

import torch
from e3nn import o3

from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_LEN, EDGE_SIZE


class E3nn(torch.nn.Module):
    """Scalar readout with learned radial tensor-product weights.

    Radial dependence works even at lmax=0. The original linear scalar readout
    is retained; higher-l hidden features do not contribute to that readout.
    """

    def __init__(
        self, cube_type: str, cube_size: int, input_level: int, lmax: int
    ):
        super().__init__()
        self.input_level = input_level
        self.cube_type = cube_type
        self.cube_size = cube_size
        self.lmax = lmax

        if self.cube_type != "cube":
            raise NotImplementedError("Only cube type is implemented.")

        axis = (
            torch.arange(EDGE_SIZE, dtype=torch.float64, device="cpu") - CUBE_MIDDLE
        ) * EDGE_LEN
        edge_vec = torch.stack(
            torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
        ).reshape(EDGE_SIZE**3, 3)
        radii = torch.linalg.vector_norm(edge_vec, dim=-1, keepdim=True)

        irreps_input = o3.Irreps(f"{self.input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join(f"{self.input_level}x{i}e" for i in range(self.lmax + 1))
        )
        irreps_output = o3.Irreps(f"{self.cube_size}x0e")
        irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        sh = o3.spherical_harmonics(
            irreps_sh, edge_vec, normalize=True, normalization="component"
        )

        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_input,
            irreps_sh,
            hidden_irreps,
            shared_weights=False,
            internal_weights=False,
        )
        self.tp1.to(device="cpu", dtype=torch.float64)
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(1, 16, device="cpu", dtype=torch.float64),
            torch.nn.SiLU(),
            torch.nn.Linear(
                16, cast(int, self.tp1.weight_numel), device="cpu", dtype=torch.float64
            ),
        )
        self.readout = o3.Linear(hidden_irreps, irreps_output).to(
            device="cpu", dtype=torch.float64
        )

        self.register_buffer("edge_vec", edge_vec, persistent=False)
        self.register_buffer("radii", radii, persistent=False)
        self.register_buffer("sh", sh, persistent=False)

    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        """Map [..., EDGE_SIZE**3, input_level] to [..., 1, cube_size]."""
        weights = self.radial(self.radii)
        f_hidden = self.tp1(f_in, self.sh, weights)
        f_hidden = f_hidden.mean(dim=-2, keepdim=True)
        return self.readout(f_hidden)
