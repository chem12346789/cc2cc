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

        self.predictor = Extractor(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=3,
            qkv_bias=False,
            num_heads=1,
            mlp_ratio=1,
            drop_rate=0,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=9,
            drop_rate=0,
            if_skip_connection_dense=True,
            dense_actv="gelu",
        )

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

        self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 2)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # Extract the central values for each channel
        b3lyp_ene = (
            0.08 * x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.19 * x[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.72 * x[:, [2], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.81 * x[:, [3], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        )
        # b3lyp_ene = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        x_center = x[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        # do mixing x and x_center using Mixture of experts mechanism
        weight_out = self.mixing_weight(x.reshape(-1, 4 * CUBE_SIZE**3))
        weight_out = torch.softmax(weight_out, dim=-1)

        # SHAPE x = (batch, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        x = x.reshape(-1, 4, CUBE_SIZE**3)
        # SHAPE x = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x = self.predictor(x)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)

        # SHAPE x = (batch, 4, CUBE_SIZE**3)
        x = x.reshape(-1, 4 * CUBE_SIZE**3)
        # SHAPE x = (batch, 4 * CUBE_SIZE**3)
        x = self.densenet(x)
        # SHAPE x = (batch, 1)

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

        mixed_output = weight_out[:, [0]] * x + weight_out[:, [1]] * x_center
        return b3lyp_ene * mixed_output
