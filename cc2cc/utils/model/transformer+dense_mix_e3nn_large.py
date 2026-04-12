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
        self.lmax = 2
        self.out_l = 0
        self.e3nn_level = 6

        self.conv1 = E3nn(
            self.cube_type, self.cube_size, self.input_level, self.lmax, self.out_l
        )
        self.conv2 = E3nn(
            self.cube_type, self.cube_size, self.input_level, self.lmax, self.out_l
        )
        self.conv3 = E3nn(
            self.cube_type, self.cube_size, self.input_level, self.lmax, self.out_l
        )
        self.conv4 = E3nn(
            self.cube_type, self.cube_size, self.input_level, self.lmax, self.out_l
        )
        self.conv5 = E3nn(
            self.cube_type, self.cube_size, self.input_level, self.lmax, self.out_l
        )
        self.conv6 = E3nn(
            self.cube_type, self.cube_size, self.input_level, self.lmax, self.out_l
        )

        self.predictor = Transformer(
            d_model=self.cube_size,
            seq_len=self.e3nn_level,
            num_layer=7,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=self.e3nn_level * self.cube_size,
            mlp=108,
            depth=7,
            dense_bias=False,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=self.input_level,
            mlp=108,
            depth=7,
            dense_bias=False,
            if_skip_connection_dense=1,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.mixing_weight = torch.nn.Linear(self.e3nn_level * self.cube_size, 6)
        self.weight_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x_center = x[:, :, self.cube_middle]

        x_in = x.permute(0, 2, 1).contiguous()
        out1 = torch.vmap(self.conv1)(x_in)
        out2 = torch.vmap(self.conv2)(x_in)
        out3 = torch.vmap(self.conv3)(x_in)
        out4 = torch.vmap(self.conv4)(x_in)
        out5 = torch.vmap(self.conv5)(x_in)
        out6 = torch.vmap(self.conv6)(x_in)
        x_cube = torch.cat([out1, out2, out3, out4, out5, out6], dim=-2)

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(
            x_cube.reshape(-1, self.e3nn_level * self.cube_size)
        )
        weight_out = self.weight_softmax(weight_out)

        x_cube = self.predictor(x_cube)
        x_cube = x_cube.reshape(-1, self.e3nn_level * self.cube_size)
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
