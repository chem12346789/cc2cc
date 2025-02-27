"""
This module contains the model class for the model used in the project.
"""

import os
import importlib.resources
from pathlib import Path

import torch
import torch.nn as nn

from cc2cc.utils.model.transformer import PredictorSmall
from cc2cc.utils.model.ATTUNet import U_Net, R2U_Net, AttU_Net
from cc2cc.utils.model.unet import UNet

ESP = torch.finfo(torch.float32).eps


class Model(nn.Module):
    """
    TODO
    Documentation for a class.
    """

    def __init__(
        self,
        input_channels=4,
        hidden_channels=64,
        output_channels=1,
        num_layers=3,
        residual=-1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.residual = residual
        self.num_layers = num_layers

        # print all contain in this file, for debugging and logging
        with importlib.resources.files("cc2cc").joinpath(
            "utils/model"
        ) as resource_path:
            file_path = Path(os.fspath(resource_path)) / "model.py"
            with open(file_path, "r", encoding="utf-8") as finput:
                print(f"#INFO: **** input file is {file_path} ****\n")
                print(finput.read())
                print("#INFO: ****************** input file end ******************\n")
                print("\n")
                print("\n")

        if self.residual < 10:
            self.model = UNet(
                self.input_channels,
                self.hidden_channels,
                self.output_channels,
                self.num_layers,
                self.residual,
            )
        elif self.residual == 10:
            self.model = U_Net()
        else:
            self.model = R2U_Net()

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        t = x[:, [-1], :, :]

        # x = x / (t + ESP)
        x = self.model(x[:, :-1, :, :])
        x = x * t

        return x
