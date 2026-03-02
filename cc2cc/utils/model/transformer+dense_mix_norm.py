"""
Generate list of model.
"""

import torch
from torch import nn

from cc2cc.utils.env_var import EDGE_SIZE, CUBE_MIDDLE
from cc2cc.utils.model.model_utils import Transformer, DenseNet

ESP = 1e-12


class Model(nn.Module):
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

        self.mixing_weight = nn.Linear(self.input_level * self.cube_size, 6)
        self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x_norm_factor = torch.sum(torch.abs(x[:, :, :]), dim=(1, 2))
        x = torch.einsum("x,x...->x...", 1 / (x_norm_factor + ESP), x)

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(
            x.reshape(-1, self.input_level * self.cube_size)
        )
        weight_out = self.weight_softmax(weight_out)

        x_cube = self.predictor(x)
        x_cube = x_cube.reshape(-1, self.input_level * self.cube_size)
        x_cube = self.densenet(x_cube)

        # # Extract the central values for each channel
        x_center = x[:, :, self.cube_middle]
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
        return torch.einsum("x,x...->x...", (x_norm_factor + ESP), mixed_output)
