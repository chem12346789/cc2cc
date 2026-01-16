"""
Generate list of model.
"""

from torch import nn
import torch

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE
from cc2cc.utils.model.model_utils import Transformer, DenseNet

ESP = 1e-8


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "cube"
        self.input_level = 4

        self.predictor = Transformer(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=4 * CUBE_SIZE**3,
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

        self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 2)
        self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """

        x_norm_factor = torch.sum(
            torch.abs(x[:, :, :, :, :]),
            dim=(1, 2, 3, 4),
            keepdim=True,
        )
        x_norm = torch.einsum(
            "x,x...->x...",
            1 / (x_norm_factor + ESP),
            x,
        )

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(x_norm.reshape(-1, 4 * CUBE_SIZE**3))
        weight_out = self.weight_softmax(weight_out)

        x_cube = x_norm.reshape(-1, 4, CUBE_SIZE**3)
        x_cube = self.predictor(x_cube)
        x_cube = x_cube.reshape(-1, 4 * CUBE_SIZE**3)
        x_cube = self.densenet(x_cube)

        # # Extract the central values for each channel
        x_center = x_norm[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        x_center = x_center.reshape(-1, 4 * 1)
        x_center = self.densenet_center(x_center)

        mixed_output = weight_out[:, [0]] * x_cube + weight_out[:, [1]] * x_center
        mixed_output = torch.einsum(
            "x,x...->x...",
            (x_norm_factor + ESP),
            mixed_output,
        )
        return mixed_output * x_norm_factor
