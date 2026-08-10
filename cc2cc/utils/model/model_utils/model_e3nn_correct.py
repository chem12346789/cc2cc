import torch
from e3nn import o3
from e3nn.o3 import Norm

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

# E3NN-based octahedral equivariant network for 3D scalar field prediction.

# This module implements an E3NN network that processes 27 points in a cube,
# each with 1 scalar channel, and outputs 27 scalars at the center point.
# The network is equivariant to full octahedral symmetry.

# Input coordinates: 27 points in a cube from (-0.01, -0.01, -0.01) to (0.01, 0.01, 0.01)
# Output coordinates: 1 point at (0, 0, 0)
EPS = 1e-8


class E3nn(torch.nn.Module):
    def __init__(
        self,
        cube_type="cube",
        cube_size=27,
        input_level=4,
        lmax=2,
        out_l=0,
    ):
        super().__init__()

        self.input_level = input_level
        self.cube_type = cube_type
        self.cube_size = cube_size
        self.out_l = out_l
        self.lmax = lmax

        if self.cube_type == "cube":
            edge_vec = torch.zeros(
                (EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3), dtype=torch.float64
            )
            for i in range(EDGE_SIZE):
                for j in range(EDGE_SIZE):
                    for k in range(EDGE_SIZE):
                        edge_vec[i, j, k, 0] = (i - CUBE_MIDDLE) * EDGE_LEN
                        edge_vec[i, j, k, 1] = (j - CUBE_MIDDLE) * EDGE_LEN
                        edge_vec[i, j, k, 2] = (k - CUBE_MIDDLE) * EDGE_LEN
            edge_vec = edge_vec.reshape(EDGE_SIZE**3, 3)
        else:
            raise NotImplementedError("Only cube type is implemented.")

        irreps_input = o3.Irreps(f"{self.input_level}x0e")
        self.hidden_irreps = o3.Irreps(
            "+".join(
                [
                    f"{self.input_level}x{i}{'e' if i % 2 == 0 else 'o'}"
                    for i in range(self.lmax + 1)
                ]
            )
        )
        self.sl_0e = next(
            sl
            for (mul, ir), sl in zip(self.hidden_irreps, self.hidden_irreps.slices())
            if ir == o3.Irrep("0e")
        )

        irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.sh = o3.spherical_harmonics(
            irreps_sh,
            self.edge_vec,
            normalize=True,
            normalization="norm",
        )
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_input,
            irreps_sh,
            self.hidden_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        self.norm_layer = Norm(self.hidden_irreps, squared=True)

        self.output_linear = torch.nn.Linear(
            (lmax + 2) * self.input_level,
            cube_size,
            bias=False,
        )

        self.register_buffer("edge_vec", edge_vec, persistent=False)

    def forward(self, f_in):
        # f_in shape: [CUBE_SIZE**3, 4]
        f_hidden = self.tp(f_in, self.sh)
        # f_hidden shape: [CUBE_SIZE**3, (lmax+1)**2]
        f_hidden = f_hidden.sum(dim=0, keepdim=True)
        # f_hidden shape: [(lmax+1)**2]
        f_norm = self.norm_layer(f_hidden)
        f_norm = (f_norm + EPS).sqrt()
        f_out_0e = f_hidden[..., self.sl_0e]
        f_scalar = torch.cat([f_out_0e, f_norm], dim=-1)
        f_scalar = self.output_linear(f_scalar)
        return f_scalar
