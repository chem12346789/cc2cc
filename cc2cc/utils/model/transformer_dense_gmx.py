"""
Generate list of model.
"""

import torch
from torch import nn

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE
from cc2cc.utils.model.model_utils import Extractor, DenseNet


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "cube"

        self.predictor1 = Extractor(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=3,
            qkv_bias=False,
            num_heads=1,
            mlp_ratio=1,
            drop_rate=0,
            atte_actv="gelu",
        )
        self.densenet1 = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=9,
            drop_rate=0,
            if_skip_connection_dense=True,
            dense_actv="gelu",
        )

        # self.predictor2 = Extractor(
        #     d_model=CUBE_SIZE**3,
        #     seq_len=4,
        #     num_layer=3,
        #     qkv_bias=False,
        #     num_heads=1,
        #     mlp_ratio=1,
        #     drop_rate=0,
        #     atte_actv="gelu",
        # )
        # self.densenet2 = DenseNet(
        #     d_model=4 * CUBE_SIZE**3,
        #     mlp=108,
        #     depth=9,
        #     drop_rate=0,
        #     if_skip_connection_dense=True,
        #     dense_actv="gelu",
        # )

        # self.predictor3 = Extractor(
        #     d_model=CUBE_SIZE**3,
        #     seq_len=4,
        #     num_layer=3,
        #     qkv_bias=False,
        #     num_heads=1,
        #     mlp_ratio=1,
        #     drop_rate=0,
        #     atte_actv="gelu",
        # )
        # self.densenet3 = DenseNet(
        #     d_model=4 * CUBE_SIZE**3,
        #     mlp=108,
        #     depth=9,
        #     drop_rate=0,
        #     if_skip_connection_dense=True,
        #     dense_actv="gelu",
        # )

        self.predictor_center = Extractor(
            d_model=1,
            seq_len=4,
            num_layer=7,
            qkv_bias=False,
            num_heads=1,
            mlp_ratio=1,
            drop_rate=0,
            atte_actv="gelu",
        )
        self.densenet_center = DenseNet(
            d_model=4,
            mlp=128,
            depth=9,
            if_skip_connection_dense=True,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.weight_softmax = nn.Softmax(dim=-1)
        self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 2)

    def forward(self, x_in):
        """
        Standard forward function, required for all nn.Module classes
        """
        # Extract the central values for each channel
        b3lyp_ene = (
            0.08 * x_in[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.19 * x_in[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.72 * x_in[:, [2], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.81 * x_in[:, [3], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        )
        # b3lyp_ene = x_in[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        x_center = x_in[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(x_in.reshape(-1, 4 * CUBE_SIZE**3))
        weight_out = self.weight_softmax(weight_out)

        # SHAPE x_in = (batch, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        x_in = x_in.reshape(-1, 4, CUBE_SIZE**3)
        # SHAPE x1 = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)

        x1 = self.predictor1(x_in)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        # SHAPE x1 = (batch, 4, CUBE_SIZE**3)
        x1 = x1.reshape(-1, 4 * CUBE_SIZE**3)
        # SHAPE x1 = (batch, 4 * CUBE_SIZE**3)
        x1 = self.densenet1(x1)
        # SHAPE x1 = (batch, 1)

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

        mixed_output = weight_out[:, [0]] * x1 + weight_out[:, [1]] * x_center
        return b3lyp_ene * mixed_output
