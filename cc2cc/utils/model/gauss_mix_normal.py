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

        self.model_type = "cube"

        self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 4)
        self.weight_softmax = nn.Softmax(dim=-1)

        self.densenet = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=4 * CUBE_SIZE**3,
            depth=2,
            out=4,
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
        weight_out = self.mixing_weight(x.reshape(-1, 4 * CUBE_SIZE**3))
        weight_out = self.weight_softmax(weight_out)

        x = self.densenet(x.reshape(-1, 4 * CUBE_SIZE**3))

        mixed_output = (
            weight_out[:, [0]] * x[:, [0]]
            + weight_out[:, [1]] * x[:, [1]]
            + weight_out[:, [2]] * x[:, [2]]
            + weight_out[:, [3]] * x[:, [3]]
        )

        return mixed_output
