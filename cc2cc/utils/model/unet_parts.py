""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        norm_layer="BatchNorm2d",
        affine=True,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if norm_layer == "BatchNorm2d":
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=False),
                nn.ReLU(),
            )
        elif norm_layer == "NoNorm2d":
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"norm_layer {norm_layer} not recognized")

    def forward(self, x):
        """Forward pass"""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer="BatchNorm2d",
        affine=True,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                norm_layer=norm_layer,
                affine=affine,
            ),
        )

    def forward(self, x):
        """Forward pass"""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer="BatchNorm2d",
        affine=True,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            norm_layer=norm_layer,
            affine=affine,
        )

    def forward(self, x1, x2):
        """Forward pass"""
        x1 = self.up(x1)
        # input is CHW
        diff_x = x2.size()[3] - x1.size()[3]
        diff_y = x2.size()[2] - x1.size()[2]

        pad_list = (
            diff_x // 2,
            diff_x - diff_x // 2,
            diff_y // 2,
            diff_y - diff_y // 2,
        )
        x1 = F.pad(x1, pad_list, mode="reflect")
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        """Forward pass"""
        return self.conv(x)
