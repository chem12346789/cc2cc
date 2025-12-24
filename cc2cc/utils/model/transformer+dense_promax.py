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

        self.predictor = Transformer(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=7,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=7,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )

        self.predictor_center = Transformer(
            d_model=1,
            seq_len=4,
            num_layer=7,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=4,
            mlp=108,
            depth=7,
            if_skip_connection_dense=1,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 2)
        self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # # Extract the central values for each channel
        x_center = x[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(x.reshape(-1, 4 * CUBE_SIZE**3))
        weight_out = self.weight_softmax(weight_out)

        # SHAPE x_cube = (batch, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        x_cube = x.reshape(-1, 4, CUBE_SIZE**3)
        # SHAPE x_cube = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x_cube = self.predictor(x_cube)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)

        # SHAPE x_cube = (batch, 4, CUBE_SIZE**3)
        x_cube = x.reshape(-1, 4 * CUBE_SIZE**3)
        # SHAPE x_cube = (batch, 4 * CUBE_SIZE**3)
        x_cube = self.densenet(x_cube)
        # SHAPE x_cube = (batch, 1)

        # SHAPE x_center = (batch, 4)
        x_center = x_center.reshape(-1, 4, 1)
        # SHAPE x_center = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x_center = self.predictor_center(x_center)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)

        # SHAPE x_center = (batch, 4, 1)
        x_center = x_center.reshape(-1, 4 * 1)
        # SHAPE x_center = (batch, 4 * 1)
        x_center = self.densenet_center(x_center)
        # SHAPE x_center = (batch, 1)

        mixed_output = weight_out[:, [0]] * x_cube + weight_out[:, [1]] * x_center
        return mixed_output
