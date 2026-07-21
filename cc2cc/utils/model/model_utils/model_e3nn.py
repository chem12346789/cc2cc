import torch
from e3nn import o3

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

# E3NN-based octahedral equivariant network for 3D scalar field prediction.

# This module implements an E3NN network that processes 27 points in a cube,
# each with 1 scalar channel, and outputs 27 scalars at the center point.
# The network is equivariant to full octahedral symmetry.

# Input coordinates: 27 points in a cube from (-0.01, -0.01, -0.01) to (0.01, 0.01, 0.01)
# Output coordinates: 1 point at (0, 0, 0)


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
            if torch.cuda.is_available():
                edge_vec = torch.zeros(
                    (EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
                    device="cuda",
                    dtype=torch.float64,
                )
            else:
                edge_vec = torch.zeros(
                    (EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
                    device="cpu",
                    dtype=torch.float64,
                )
            self.register_buffer("edge_vec", edge_vec, persistent=False)
            for i in range(EDGE_SIZE):
                for j in range(EDGE_SIZE):
                    for k in range(EDGE_SIZE):
                        self.edge_vec[i, j, k, 0] = (i - CUBE_MIDDLE) * EDGE_LEN
                        self.edge_vec[i, j, k, 1] = (j - CUBE_MIDDLE) * EDGE_LEN
                        self.edge_vec[i, j, k, 2] = (k - CUBE_MIDDLE) * EDGE_LEN
        else:
            raise NotImplementedError("Only cube type is implemented.")
        self.edge_vec = self.edge_vec.reshape(EDGE_SIZE**3, 3)

        if torch.cuda.is_available():
            center_vec = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
        else:
            center_vec = torch.zeros(1, 3, device="cpu", dtype=torch.float64)
        self.register_buffer("center_vec", center_vec, persistent=False)
        self.center_vec = self.center_vec.reshape(1, 3)

        irreps_input = o3.Irreps(f"{self.input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join([f"{self.input_level}x{i}e" for i in range(self.lmax + 1)])
        )
        irreps_output = o3.Irreps(f"{self.cube_size}x{self.out_l}e")

        irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.sh = o3.spherical_harmonics(
            irreps_sh,
            self.edge_vec,
            normalize=False,
            normalization="norm",
        )
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_input,
            irreps_sh,
            hidden_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        irreps_sh_center = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.sh_center = o3.spherical_harmonics(
            irreps_sh_center,
            self.center_vec,
            normalize=False,
            normalization="norm",
        )
        self.tp_center = o3.FullyConnectedTensorProduct(
            hidden_irreps,
            irreps_sh_center,
            irreps_output,
            shared_weights=True,
            internal_weights=True,
        )

        # xavier_uniform_ initialization for the tensor product weights
        with torch.no_grad():
            for weight in self.tp.weight_views():
                mul_1, mul_2, mul_out = weight.shape
                # formula from torch.nn.init.xavier_uniform_
                a = (6 / (mul_1 * mul_2 + mul_out)) ** 0.5
                new_weight = torch.empty_like(weight)
                new_weight.uniform_(-a, a)
                weight[:] = new_weight
            for weight in self.tp_center.weight_views():
                mul_1, mul_2, mul_out = weight.shape
                # formula from torch.nn.init.xavier_uniform_
                a = (6 / (mul_1 * mul_2 + mul_out)) ** 0.5
                new_weight = torch.empty_like(weight)
                new_weight.uniform_(-a, a)
                weight[:] = new_weight

    def forward(self, f_in):
        # f_in shape: [CUBE_SIZE**3, 4]
        f_hidden = self.tp(f_in, self.sh)
        # f_hidden shape: [CUBE_SIZE**3, (lmax+1)**2]
        f_hidden = f_hidden.sum(dim=0)
        # f_hidden shape: [(lmax+1)**2]
        f_out = self.tp_center(f_hidden, self.sh_center)
        # f_out shape: [CUBE_SIZE**3, 1]
        return f_out
