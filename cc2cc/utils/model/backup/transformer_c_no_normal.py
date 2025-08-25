"""
Generate list of model.
"""

import torch
from torch import nn

from cc2cc.utils.model.model_utils import Extractor, DenseNet


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "center_4"

        self.predictor = Extractor(
            d_model=1,
            seq_len=4,
            num_layer=7,
            qkv_bias=False,
            num_heads=1,
            mlp_ratio=1,
            drop_rate=0,
            atte_actv="gelu",
            atte_normal="rms",
        )

        self.densenet = DenseNet(
            d_model=4,
            mlp=128,
            depth=9,
            if_skip_connection_dense=1,
            drop_rate=0,
            dense_actv="gelu",
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # Extract the central values for each channel
        b3lyp_ene = (
            0.08 * x[:, [0]] + 0.19 * x[:, [1]] + 0.72 * x[:, [2]] + 0.81 * x[:, [3]]
        )
        # b3lyp_ene = x[:, [0]]

        # SHAPE x = (batch, 4)
        x = x.reshape(-1, 4, 1)
        # SHAPE x = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x = self.predictor(x)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)

        # SHAPE x = (batch, 4, 1)
        x = x.reshape(-1, 4 * 1)
        # SHAPE x = (batch, 4 * 1)
        x = self.densenet(x)
        # SHAPE x = (batch, 1)
        x = x * b3lyp_ene
        return x
