"""
Generate list of model.
"""

from torch import nn

from cc2cc.utils.env_var import EDGE_SIZE, CUBE_MIDDLE
from cc2cc.utils.model.model_utils import Transformer, DenseNet


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

        self.densenet = DenseNet(
            d_model=4 * EDGE_SIZE**3,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=4,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.mixing_weight = nn.Linear(4 * EDGE_SIZE**3, 2)
        self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(
            x.reshape(-1, self.input_level * self.cube_size)
        )
        weight_out = self.weight_softmax(weight_out)

        x_cube = x.reshape(-1, self.input_level * self.cube_size)
        x_cube = self.densenet(x_cube)

        # # Extract the central values for each channel
        x_center = x[:, :, self.cube_middle]
        x_center = x_center.reshape(-1, 4 * 1)
        x_center = self.densenet_center(x_center)

        mixed_output = weight_out[:, [0]] * x_cube + weight_out[:, [1]] * x_center
        return mixed_output
