"""Distance- and angle-aware scalar readout on a cubic grid.

Learned radial weights use distances normalized by EDGE_LEN. TensorSquare
couples pooled angular features into invariants alongside linear scalar features.
A nonlinear scalar head merges both paths without breaking O(3) invariance.
The constructor and output shapes are unchanged; earlier linear-mixer checkpoints
are incompatible.
"""

from typing import cast

import torch
from e3nn import o3

from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_LEN, EDGE_SIZE


class E3nn(torch.nn.Module):
    """Scalar readout with learned radial weights and angular invariants.

    Radial dependence works even at lmax=0. For lmax>0, TensorSquare
    couples pooled spherical harmonics to retain relative-angle dependence.
    """

    def __init__(self, cube_type: str, cube_size: int, input_level: int, lmax: int):
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
        radii = torch.linalg.vector_norm(edge_vec, dim=-1, keepdim=True) / EDGE_LEN

        irreps_input = o3.Irreps(f"{self.input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join(
                f"{self.input_level}x{i}{'e' if i % 2 == 0 else 'o'}"
                for i in range(self.lmax + 1)
            )
        )
        scalar_irreps = o3.Irreps(f"{self.cube_size}x0e")
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

        self.tensor_square = o3.TensorSquare(
            irreps_in=hidden_irreps,
            irreps_out=scalar_irreps,
            irrep_normalization="component",
        )
        self.tensor_square.to(device="cpu", dtype=torch.float64)

        self.readout = o3.Linear(hidden_irreps, scalar_irreps)
        self.readout = self.readout.to(device="cpu", dtype=torch.float64)
        self.mixer = torch.nn.Sequential(
            torch.nn.Linear(
                2 * self.cube_size, self.cube_size, device="cpu", dtype=torch.float64
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                self.cube_size, self.cube_size, device="cpu", dtype=torch.float64
            ),
        )

        self.register_buffer("edge_vec", edge_vec, persistent=False)
        self.register_buffer("radii", radii, persistent=False)
        self.register_buffer("sh", sh, persistent=False)

    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        """Map [..., EDGE_SIZE**3, input_level] to [..., 1, cube_size]."""
        weights = self.radial(self.radii)
        f_hidden = self.tp1(f_in, self.sh, weights)
        f_hidden = f_hidden.mean(dim=-2, keepdim=True)
        f_quad = self.tensor_square(f_hidden)
        f_lin = self.readout(f_hidden)
        return self.mixer(torch.cat([f_quad, f_lin], dim=-1))
