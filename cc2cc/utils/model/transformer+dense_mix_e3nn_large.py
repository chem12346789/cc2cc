"""
Generate list of model.
"""

import torch

from cc2cc.utils.env_var import EDGE_SIZE
from cc2cc.utils.model.model_utils import Transformer, DenseNet, E3nn


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
        self.lmax = 4
        self.out_l = 0
        self.d_model = self.cube_size * (2 * self.out_l + 1)
        self.e3nn_level = 8

        self.e3nn = torch.nn.ModuleList(
            [
                E3nn(
                    self.cube_type,
                    self.cube_size,
                    self.input_level,
                    self.lmax,
                    self.out_l,
                )
                for _ in range(self.e3nn_level)
            ]
        )

        self.predictor = Transformer(
            d_model=self.d_model,
            seq_len=self.e3nn_level,
            num_layer=7,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=self.e3nn_level * self.d_model,
            mlp=108,
            depth=7,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=self.input_level,
            mlp=108,
            depth=7,
            dense_bias=False,
            if_skip_connection_dense=0,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.mixing_weight = torch.nn.Linear(self.e3nn_level * self.d_model, 6)
        self.weight_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x_center = x[:, :, self.cube_middle]

        x_in = x.permute(0, 2, 1).contiguous()
        x_cube = []
        for e3nn_layer in self.e3nn:
            x_out = torch.vmap(e3nn_layer)(x_in)
            x_cube.append(x_out)
        x_cube = torch.cat(x_cube, dim=-2)
        print(f"x_cube shape: {x_cube.shape}")

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(
            x_cube.reshape(-1, self.e3nn_level * self.d_model)
        )
        weight_out = self.weight_softmax(weight_out)

        x_cube = self.predictor(x_cube)
        x_cube = x_cube.reshape(-1, self.e3nn_level * self.d_model)
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
