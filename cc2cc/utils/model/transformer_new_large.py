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
            d_model=4,
            seq_len=CUBE_SIZE**3,
            num_layer=7,
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
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )

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

        # SHAPE x = (batch, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        x = x.reshape(-1, 4, CUBE_SIZE**3)
        x = torch.permute(x, (0, 2, 1))
        # SHAPE x = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x = self.predictor(x)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x = torch.permute(x, (0, 2, 1))

        # SHAPE x = (batch, 4, CUBE_SIZE**3)
        x = x.reshape(-1, 4 * CUBE_SIZE**3)
        # SHAPE x = (batch, 4 * CUBE_SIZE**3)
        x = self.densenet(x)
        # SHAPE x = (batch, 1)
        x = b3lyp_ene * x
        return x
