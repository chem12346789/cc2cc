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
        self.input_level = 4

        self.predictor1 = Transformer(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )
        self.densenet1 = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.predictor2 = Transformer(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )
        self.densenet2 = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.predictor3 = Transformer(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )
        self.densenet3 = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.predictor4 = Transformer(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )
        self.densenet4 = DenseNet(
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

        self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 10)
        self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(x.reshape(-1, 4 * CUBE_SIZE**3))
        weight_out = self.weight_softmax(weight_out)

        x_cube_den = x.reshape(-1, 4 * CUBE_SIZE**3)
        x_cube_den = self.densenet1(x_cube_den)

        x_cube1 = x.reshape(-1, 4, CUBE_SIZE**3)
        x_cube1 = self.predictor1(x_cube1)
        x_cube1 = x_cube1.reshape(-1, 4 * CUBE_SIZE**3)
        x_cube1 = self.densenet1(x_cube1)

        x_cube2 = x.reshape(-1, 4, CUBE_SIZE**3)
        x_cube2 = self.predictor2(x_cube2)
        x_cube2 = x_cube2.reshape(-1, 4 * CUBE_SIZE**3)
        x_cube2 = self.densenet2(x_cube2)

        x_cube3 = x.reshape(-1, 4, CUBE_SIZE**3)
        x_cube3 = self.predictor3(x_cube3)
        x_cube3 = x_cube3.reshape(-1, 4 * CUBE_SIZE**3)
        x_cube3 = self.densenet3(x_cube3)

        x_cube4 = x.reshape(-1, 4, CUBE_SIZE**3)
        x_cube4 = self.predictor4(x_cube4)
        x_cube4 = x_cube4.reshape(-1, 4 * CUBE_SIZE**3)
        x_cube4 = self.densenet4(x_cube4)

        # # Extract the central values for each channel
        x_center = x[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        x_center = x_center.reshape(-1, 4 * 1)
        x_center = self.densenet_center(x_center)

        mixed_output = (
            weight_out[:, [0]] * x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + weight_out[:, [1]] * x[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + weight_out[:, [2]] * x[:, [2], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + weight_out[:, [3]] * x[:, [3], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + weight_out[:, [4]] * x_center
            + weight_out[:, [5]] * x_cube_den
            + weight_out[:, [6]] * x_cube1
            + weight_out[:, [7]] * x_cube2
            + weight_out[:, [8]] * x_cube3
            + weight_out[:, [9]] * x_cube4
        )
        return mixed_output
