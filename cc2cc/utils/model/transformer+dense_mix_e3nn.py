"""
Generate list of model.
"""

import torch

from e3nn import o3, nn

from cc2cc.utils.model.model_utils import Transformer, DenseNet
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


class Model(torch.nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.cube_type = "cube"
        self.cube_size = EDGE_SIZE**3
        self.cube_middle = (self.cube_size - 1) // 2
        self.input_level = 4
        self.before_weight = False

        self.E3nn = E3nn(
            cube_type=self.cube_type,
            cube_size=self.cube_size,
            input_level=self.input_level,
        )

        self.predictor = Transformer(
            d_model=self.cube_size,
            seq_len=self.input_level,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=self.input_level * self.cube_size,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=self.input_level,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.mixing_weight = torch.nn.Linear(self.input_level * self.cube_size, 6)
        self.weight_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x_center = x[:, :, self.cube_middle]

        x_cube = self.E3nn(x)

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(
            x_cube.reshape(-1, self.input_level * self.cube_size)
        )
        weight_out = self.weight_softmax(weight_out)

        x_cube = self.predictor(x_cube)
        x_cube = x_cube.reshape(-1, self.input_level * self.cube_size)
        x_cube = self.densenet(x_cube)

        # # Extract the central values for each channel
        x_center = x_center.reshape(-1, self.input_level)
        x_center = self.densenet_center(x_center)

        mixed_output = (
            weight_out[:, [0]] * x_cube
            + weight_out[:, [1]] * x_center
            + weight_out[:, [2]] * x[:, [0], self.cube_middle]
            + weight_out[:, [3]] * x[:, [1], self.cube_middle]
            + weight_out[:, [4]] * x[:, [2], self.cube_middle]
            + weight_out[:, [5]] * x[:, [3], self.cube_middle]
        )
        return mixed_output
