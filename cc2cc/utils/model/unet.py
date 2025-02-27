import torch
import torch.nn as nn

from cc2cc.utils.model.unet_parts import DoubleConv, Down, Up, OutConv

ESP = torch.finfo(torch.float32).eps


class UNet(nn.Module):
    """
    TODO
    Documentation for a class.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        num_layers,
        residual,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.residual = residual
        self.num_layers = num_layers

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
                    self.hidden_channels * 2 ** (i),
                    norm_layer=norm_layer,
                    affine=affine,
                )
                for i in range(self.num_layers)[::-1]
            ]
        )
        self.outc = OutConv(self.hidden_channels, self.output_channels)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """

        x = self.inc(x)
        x_down = []
        for down in self.down_layers:
            x_down.append(x)
            x = down(x)
        for i, up in enumerate(self.up_layers):
            x = up(x, x_down[-i - 1])
        x = self.outc(x)
        return x
