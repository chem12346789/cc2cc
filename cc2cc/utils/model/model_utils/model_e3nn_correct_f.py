import torch
import numpy as np
from e3nn import o3

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

# E3NN-based octahedral equivariant network for 3D scalar field prediction.

# This module implements an E3NN network that processes 27 points in a cube,
# each with 1 scalar channel, and outputs 27 scalars at the center point.
# The network is equivariant to full octahedral symmetry.

# Input coordinates: 27 points in a cube from (-0.01, -0.01, -0.01) to (0.01, 0.01, 0.01)
# Output: 27 invariants at the center point.


class E3nn(torch.nn.Module):

    def __init__(self, cube_type, cube_size, input_level, lmax):
        super().__init__()

        self.input_level = input_level
        self.cube_type = cube_type
        self.cube_size = cube_size
        self.lmax = lmax

        if self.cube_type == "cube":
            edge_vec = torch.zeros(
                (EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
                device="cpu",
                dtype=torch.get_default_dtype(),
            )
            for i in range(EDGE_SIZE):
                for j in range(EDGE_SIZE):
                    for k in range(EDGE_SIZE):
                        edge_vec[i, j, k, 0] = (i - CUBE_MIDDLE) * EDGE_LEN
                        edge_vec[i, j, k, 1] = (j - CUBE_MIDDLE) * EDGE_LEN
                        edge_vec[i, j, k, 2] = (k - CUBE_MIDDLE) * EDGE_LEN
        else:
            raise NotImplementedError("Only cube type is implemented.")
        edge_vec = edge_vec.reshape(EDGE_SIZE**3, 3)

        irreps_input = o3.Irreps(f"{self.input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join([f"{self.input_level}x{i}e" for i in range(self.lmax + 1)])
        )
        # hidden_irreps = o3.Irreps(
        #     "+".join(
        #         [
        #             f"{self.input_level}x{i}{'e' if i % 2 == 0 else 'o'}"
        #             for i in range(self.lmax + 1)
        #         ]
        #     )
        # )
        irreps_output = o3.Irreps(f"{self.cube_size}x{0}e")

        irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        sh = o3.spherical_harmonics(
            irreps_sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_input,
            irreps_sh,
            hidden_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        self.tensor_square = o3.TensorSquare(
            irreps_in=hidden_irreps,
            irreps_out=irreps_output,
            irrep_normalization="component",
        )

        # uniform_ initialization for the tensor product weights
        with torch.no_grad():
            self.tp1.weight.uniform_(-1, 1)
            # Scale higher-l paths smaller: weight in [-1/(l+1), 1/(l+1)]
            offset = 0
            for ins in self.tensor_square.instructions:
                if not ins.has_weight:
                    continue
                numel = 1
                for s in ins.path_shape:
                    numel *= s
                l = hidden_irreps[ins.i_in1].ir.l  # l1 == l2 for TensorSquare
                scale = 1.0 / (2 * l + 1)
                self.tensor_square.weight[offset : offset + numel].uniform_(
                    -scale, scale
                )
                offset += numel

        self.register_buffer("edge_vec", edge_vec, persistent=False)
        self.register_buffer("sh", sh, persistent=False)

    def forward(self, f_in):
        # f_in shape: [CUBE_SIZE**3, 4]
        h_local = self.tp1(f_in, self.sh)
        # f_hidden shape: [CUBE_SIZE**3, (lmax+1)**2]
        h_global = h_local.sum(dim=-2, keepdim=True)
        # f_hidden shape: [(lmax+1)**2]
        f_out = self.tensor_square(h_global, None)
        # f_out shape: [CUBE_SIZE**3]
        # f_out shape: [1, CUBE_SIZE**3]
        return f_out
