import os
import importlib.resources
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cc2cc.utils.model.unet_parts import DoubleConv, Down, Up, OutConv
from cc2cc.utils.model.transformer import PredictorSmall

ESP = torch.finfo(torch.float32).eps


class Model(nn.Module):
    """
    TODO
    Documentation for a class.
    """

    def __init__(
        self,
        input_channels=4,
        hidden_channels=32,
        output_channels=1,
        num_layers=3,
        residual=0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.residual = residual
        self.num_layers = num_layers

        print(
            f"Model: UNet, residual: {self.residual} "
            f"num_layers: {self.num_layers} "
            f"hidden_channels: {self.hidden_channels} "
            f"input_channels: {self.input_channels} "
            f"output_channels: {self.output_channels} "
        )

        # print all contain in this file, for debugging and logging
        with importlib.resources.files("cc2cc").joinpath(
            "utils/model"
        ) as resource_path:
            file_path = Path(os.fspath(resource_path)) / "unet.py"
            finput = open(file_path, "r")
            print("#INFO: **** input file is %s ****\n" % file_path)
            print(finput.read())
            print("#INFO: ******************** input file end ********************\n")
            print("\n")
            print("\n")
            finput.close()

        if self.residual < 10:
            if self.residual == 0:
                norm_layer = "BatchNorm2d"
                affine = True
            elif self.residual == 1:
                norm_layer = "BatchNorm2d"
                affine = False
            else:
                norm_layer = "NoNorm2d"
                affine = True

            print(f"norm_layer: {norm_layer} affine: {affine}")

            self.inc = DoubleConv(
                self.input_channels,
                self.hidden_channels,
                norm_layer=norm_layer,
                affine=affine,
            )

            self.down_layers = nn.ModuleList(
                [
                    Down(
                        self.hidden_channels * 2 ** (i),
                        self.hidden_channels * 2 ** (i + 1),
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for i in range(self.num_layers)
                ]
            )
            self.up_layers = nn.ModuleList(
                [
                    Up(
                        self.hidden_channels * 2 ** (i + 1),
                        self.hidden_channels * 2**i,
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for i in range(self.num_layers)[::-1]
                ]
            )
            self.outc = OutConv(self.hidden_channels, self.output_channels)
        else:
            if self.residual == 10:
                self.model = PredictorSmall()

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """

        if self.residual < 10:
            t = x[:, [0], :, :]
            # x = torch.sigmoid(x)
            x = self.inc(x[:, :, :, :])
            x_down = []
            for down in self.down_layers:
                x_down.append(x)
                x = down(x)
            for i, up in enumerate(self.up_layers):
                x = up(x, x_down[-i - 1])
            return self.outc(x) * t
        else:
            t = x[:, [0], :, :]
            x = torch.sigmoid(t)
            return self.model(x) * t
