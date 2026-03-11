import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE


# E3NN-based octahedral equivariant network for 3D scalar field prediction.

# This module implements an E3NN network that processes 27 points in a cube,
# each with 1 scalar channel, and outputs 27 scalars at the center point.
# The network is equivariant to full octahedral symmetry.

# Input coordinates: 27 points in a cube from (-0.01, -0.01, -0.01) to (0.01, 0.01, 0.01)
# Output coordinates: 1 point at (0, 0, 0)


class Conv(torch.nn.Module):
    def __init__(self, cube_type="cube", cube_size=27, input_level=4, lmax=2):
        super().__init__()

        self.input_level = input_level
        self.cube_type = cube_type
        self.cube_size = cube_size
        self.lmax = lmax
        num_basis = 10
        self.lmax = 2

        if self.cube_type == "cube":
            edge_vec = torch.zeros(
                (EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
                device="cuda",
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
        center_vec = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
        self.register_buffer("center_vec", center_vec, persistent=False)
        self.edge_vec = self.edge_vec.reshape(EDGE_SIZE**3, 3)

        irreps_input = o3.Irreps(f"{self.input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join([f"{self.input_level}x{i}e" for i in range(self.lmax + 1)])
        )
        irreps_output = o3.Irreps(f"{self.cube_size}x0e")

        irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        self.sh = o3.spherical_harmonics(
            irreps_sh, self.edge_vec, normalize=True, normalization="component"
        )
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_input,
            irreps_sh,
            hidden_irreps,
            # shared_weights=False,
            shared_weights=True,
            internal_weights=True,
        )

        # xavier_uniform_ initialization for the tensor product weights
        with torch.no_grad():
            self.tp.weight.uniform_(-1, 1)

        # self.emb = soft_one_hot_linspace(
        #     self.edge_vec.norm(dim=1),
        #     -1,
        #     1,
        #     num_basis,
        #     basis="smooth_finite",
        #     cutoff=True,
        # ).mul(num_basis**0.5)
        # self.fc = nn.FullyConnectedNet(
        #     [num_basis, 16, self.tp.weight_numel], torch.relu
        # )

        self.sh_center = o3.spherical_harmonics(
            irreps_sh,
            self.center_vec,
            normalize=True,
            normalization="component",
        )
        self.tp_center = o3.FullyConnectedTensorProduct(
            hidden_irreps,
            irreps_sh,
            irreps_output,
            shared_weights=True,
            internal_weights=True,
        )

    def forward(self, f_in):
        # f_in shape: [CUBE_SIZE**3, 4]
        f_hidden = self.tp(f_in, self.sh)
        # f_hidden shape: [CUBE_SIZE**3, (lmax+1)**2]
        f_hidden = f_hidden.sum(dim=0)
        # f_hidden shape: [(lmax+1)**2]
        f_out = self.tp_center(f_hidden, self.sh_center)
        # f_out shape: [27]
        return f_out


class E3nn(torch.nn.Module):

    def __init__(self, cube_type="cube", cube_size=27, input_level=4, lmax=2):
        super().__init__()

        self.cube_type = cube_type
        self.cube_size = cube_size
        self.input_level = input_level
        self.lmax = lmax

        self.conv1 = Conv(self.cube_type, self.cube_size, self.input_level, self.lmax)
        self.conv2 = Conv(self.cube_type, self.cube_size, self.input_level, self.lmax)
        self.conv3 = Conv(self.cube_type, self.cube_size, self.input_level, self.lmax)
        self.conv4 = Conv(self.cube_type, self.cube_size, self.input_level, self.lmax)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        out1 = torch.vmap(self.conv1)(x)
        out2 = torch.vmap(self.conv2)(x)
        out3 = torch.vmap(self.conv3)(x)
        out4 = torch.vmap(self.conv4)(x)
        x_cube = torch.cat([out1, out2, out3, out4], dim=-2)
        return x_cube
