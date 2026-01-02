"""
Generate list of model.
"""

import torch
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

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        b3lyp_ene = (
            0.08 * x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.19 * x[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.72 * x[:, [2], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.81 * x[:, [3], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        )
        return b3lyp_ene
