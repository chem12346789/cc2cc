"""
Generate list of model.
"""

from torch import nn

from cc2cc.utils.model.model_utils import Transformer, DenseNet


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "cube5"
        self.input_level = 4
        self.cube_size = 5
        self.cube_middle = 2

        self.predictor = Transformer(
            d_model=5,
            seq_len=4,
            num_layer=3,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=4 * 5,
            mlp=108,
            depth=3,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=4,
            mlp=108,
            depth=3,
            dense_bias=False,
            if_skip_connection_dense=0,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.mixing_weight = nn.Linear(4 * 5, 6)
        self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(x.reshape(-1, 4 * 5))
        weight_out = self.weight_softmax(weight_out)

        x_cube = x.reshape(-1, 4, 5)
        x_cube = self.predictor(x_cube)
        x_cube = x_cube.reshape(-1, 4 * 5)
        x_cube = self.densenet(x_cube)

        # # Extract the central values for each channel
        x_center = x[:, :, self.cube_middle]
        x_center = x_center.reshape(-1, 4 * 1)
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
