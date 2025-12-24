"""
Generate list of model.
"""

from torch import nn

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE
from cc2cc.utils.model.model_utils import Transformer, DenseNet


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "cube"

        # self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 4)
        # self.weight_softmax = nn.Softmax(dim=-1)

        # self.predictor = Transformer(
        #     d_model=CUBE_SIZE**3,
        #     seq_len=4,
        #     num_layer=1,
        #     qkv_bias=False,
        #     ffn_bias=False,
        #     mlp_ratio=1,
        #     atte_actv="gelu",
        # )

        self.densenet = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=2,
            if_skip_connection_dense=1,
            dense_bias=False,
            dense_actv="gelu",
        )

    def forward(self, x_in):
        """
        Standard forward function, required for all nn.Module classes
        """
        # x_cube = x_in.reshape(-1, 4, CUBE_SIZE**3)
        # x_cube = self.predictor(x_cube)
        x_cube = x_in.reshape(-1, 4 * CUBE_SIZE**3)
        x_cube = self.densenet(x_cube)

        # mixed_output = (
        #     weight_out[:, [0]] * x_cube[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        #     + weight_out[:, [1]] * x_cube[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        #     + weight_out[:, [2]] * x_cube[:, [2], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        #     + weight_out[:, [3]] * x_cube[:, [3], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        # )

        return x_cube
