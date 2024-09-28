import torch.nn as nn

from dft2cc.utils.model.unet_parts import DoubleConv, Down, Up, OutConv


class Model(nn.Module):
    """
    TODO
    Documentation for a class.
    """

    def __init__(self):
        super().__init__()
        self.input_size = 1
        self.hidden_size = 64
        self.output_size = 1
        self.num_layers = 3
        self.residual = -1

        print(
            f"Model: UNet, residual: {self.residual}"
            f"num_layers: {self.num_layers}"
            f"hidden_size: {self.hidden_size}"
            f"input_size: {self.input_size}"
            f"output_size: {self.output_size}"
        )

        if self.residual == -1:
            norm_layer = "NoNorm2d"
            affine = True
        if self.residual == 0:
            norm_layer = "BatchNorm2d"
            affine = True
        if self.residual == 1:
            norm_layer = "BatchNorm2d"
            affine = False
        if self.residual == 2:
            norm_layer = "InstanceNorm2d"
            affine = True
        if self.residual == 3:
            norm_layer = "InstanceNorm2d"
            affine = False
        if self.residual == 4:
            norm_layer = "GroupNorm1"
            affine = True
        if self.residual == 5:
            norm_layer = "GroupNorm1"
            affine = False
        if self.residual == 6:
            norm_layer = "GroupNorm2"
            affine = True
        if self.residual == 7:
            norm_layer = "GroupNorm2"
            affine = False
        if self.residual == 8:
            norm_layer = "GroupNorm4"
            affine = True
        if self.residual == 9:
            norm_layer = "GroupNorm4"
            affine = False
        if self.residual == 10:
            norm_layer = "GroupNorm8"
            affine = True
        if self.residual == 11:
            norm_layer = "GroupNorm8"
            affine = False
        if self.residual == 12:
            norm_layer = "GroupNorm16"
            affine = True
        if self.residual == 13:
            norm_layer = "GroupNorm16"
            affine = False

        if "GroupNorm" in norm_layer:
            self.inc = DoubleConv(
                self.input_size,
                self.hidden_size,
                norm_layer="NoNorm2d",
                affine=True,
            )
        else:
            self.inc = DoubleConv(
                self.input_size,
                self.hidden_size,
                norm_layer=norm_layer,
                affine=affine,
            )

        self.down_layers = nn.ModuleList(
            [
                Down(
                    self.hidden_size * 2 ** (i),
                    self.hidden_size * 2 ** (i + 1),
                    norm_layer=norm_layer,
                    affine=affine,
                )
                for i in range(self.num_layers)
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                Up(
                    self.hidden_size * 2 ** (i + 1),
                    self.hidden_size * 2**i,
                    norm_layer=norm_layer,
                    affine=affine,
                )
                for i in range(self.num_layers)[::-1]
            ]
        )
        self.outc = OutConv(self.hidden_size, self.output_size)

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
        logits = self.outc(x)
        return logits
