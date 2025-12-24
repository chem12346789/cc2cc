"""
Generate list of model.
"""

from torch import nn
import numpy as np

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE
from cc2cc.utils.model.model_utils import Transformer, DenseNet


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "center_4"

        self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 4)
        self.weight_softmax = nn.Softmax(dim=-1)

        self.densenet1 = DenseNet(
            d_model=4,
            mlp=4 * CUBE_SIZE**3,
            depth=2,
            dense_bias=False,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )
        self.densenet2 = DenseNet(
            d_model=4,
            mlp=4 * CUBE_SIZE**3,
            depth=2,
            dense_bias=False,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )
        self.densenet3 = DenseNet(
            d_model=4,
            mlp=4 * CUBE_SIZE**3,
            depth=2,
            dense_bias=False,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )
        self.densenet4 = DenseNet(
            d_model=4,
            mlp=4 * CUBE_SIZE**3,
            depth=2,
            dense_bias=False,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # Extract the central values for each channel
        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(x)
        weight_out = self.weight_softmax(weight_out)

        x1 = self.densenet1(x)
        x2 = self.densenet2(x)
        x3 = self.densenet3(x)
        x4 = self.densenet4(x)

        mixed_output = (
            weight_out[:, [0]] * x1
            + weight_out[:, [1]] * x2
            + weight_out[:, [2]] * x3
            + weight_out[:, [3]] * x4
        )

        return mixed_output
