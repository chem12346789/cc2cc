""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => LeakyReLU) * 2"""

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
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=False),
                nn.ReLU(inplace=True),
            )
        elif norm_layer == "InstanceNorm2d":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm2d(
                    mid_channels, affine=affine, track_running_stats=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm2d(
                    out_channels, affine=affine, track_running_stats=False
                ),
                nn.ReLU(inplace=True),
            )
        elif norm_layer == "GroupNorm1":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(1, mid_channels, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(1, out_channels, affine=affine),
                nn.ReLU(inplace=True),
            )
        elif norm_layer == "GroupNorm2":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(2, mid_channels, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(2, out_channels, affine=affine),
                nn.ReLU(inplace=True),
            )
        elif norm_layer == "GroupNorm4":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(4, mid_channels, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(4, out_channels, affine=affine),
                nn.ReLU(inplace=True),
            )
        elif norm_layer == "GroupNorm8":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(8, mid_channels, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(8, out_channels, affine=affine),
                nn.ReLU(inplace=True),
            )
        elif norm_layer == "GroupNorm16":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(16, mid_channels, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(16, out_channels, affine=affine),
                nn.ReLU(inplace=True),
            )
        elif norm_layer == "NoNorm2d":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"norm_layer {norm_layer} not recognized")

    def forward(self, x):
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
            DoubleConv(in_channels, out_channels, norm_layer=norm_layer, affine=affine),
        )

    def forward(self, x):
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
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            in_channels, out_channels, norm_layer=norm_layer, affine=affine
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        pad_list = (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        x1 = F.pad(x1, pad_list, mode="reflect")
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
