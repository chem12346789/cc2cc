"""
Generate list of model.
"""

from torch import nn
import torch

from cc2cc.utils.env_var import EDGE_SIZE
from cc2cc.utils.model.model_utils import Transformer, DenseNet


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.cube_type = "cube"
        self.cube_size = 27
        self.cube_middle = (self.cube_size - 1) // 2
        self.input_level = 4
        self.before_weight = False

        self.predictor = Transformer(
            d_model=4,
            seq_len=self.input_level,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=self.input_level * 4,
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

        self.mixing_weight = nn.Linear(self.input_level * 4, 5)
        self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        num_of_batch = x.shape[0]
        x_in = torch.stack(
            [
                x[:, :, self.cube_middle],
                x[:, :, [0, 2, 6, 8, 18, 20, 24, 26]].sum(dim=-1),
                x[:, :, [9, 3, 1, 17, 23, 25, 15, 5, 19, 11, 21, 7]].sum(dim=-1),
                x[:, :, [12, 14, 10, 16, 4, 22]].sum(dim=-1),
            ],
            dim=-1,
        )

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(
            x_in.reshape(num_of_batch, self.input_level * 4)
        )
        weight_out = self.weight_softmax(weight_out)

        x_cube = self.predictor(x_in)
        x_cube = x_cube.reshape(num_of_batch, self.input_level * 4)
        x_cube = self.densenet(x_cube)

        # # Extract the central values for each channel
        x_center = x_in[:, :, 0]
        x_center = x_center.reshape(num_of_batch, self.input_level)
        x_center = self.densenet_center(x_center)

        mixed_output = (
            +weight_out[:, [0]] * x_center
            + weight_out[:, [1]] * x[:, [0], 0]
            + weight_out[:, [2]] * x[:, [1], 0]
            + weight_out[:, [3]] * x[:, [2], 0]
            + weight_out[:, [4]] * x[:, [3], 0]
        )
        return mixed_output
