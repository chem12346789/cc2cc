"""
An 3d cnn model
"""

import importlib.resources
import os
from pathlib import Path

from torch import nn

from cc2cc.utils.env_var import CUBE_MIDDLE

D_MODEL = 108
MLP = 108
DENSE_DEPTH = 5
IF_SKIP_CONNECTION = 1

DROP_RATE = 0.0001

# DENSE_ACTV = "relu"
# DENSE_NORMAL = "layer"

DENSE_ACTV = "gelu"
DENSE_NORMAL = "rms"


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.mlp = kwargs.get("mlp", MLP)
        self.depth = kwargs.get("depth", DENSE_DEPTH)
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE)

        # print all contain in this file, for debugging and logging
        with importlib.resources.files("cc2cc").joinpath(
            "utils/model"
        ) as resource_path:
            file_path = Path(os.fspath(resource_path)) / "densenet.py"
            with open(file_path, "r", encoding="utf-8") as finput:
                print(f"#INFO: **** input file is {file_path} ****\n")
                print(finput.read())
                print("#INFO: ****************** input file end ******************\n")
                print("\n")
                print("\n")

        self.sizes = [self.d_model] + [self.mlp] * (self.depth - 1) + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(self.sizes, self.sizes[1:])
            ]
        )

        if DENSE_ACTV == "relu":
            self.actv_fn = nn.ReLU()
        elif DENSE_ACTV == "gelu":
            self.actv_fn = nn.GELU()

        if DENSE_NORMAL == "layer":
            self.norm = nn.ModuleList(
                [nn.LayerNorm(i_size) for i_size in self.sizes[:-2]]
            )
        elif DENSE_NORMAL == "rms":
            self.norm = nn.ModuleList(
                [nn.RMSNorm(i_size) for i_size in self.sizes[:-2]]
            )

        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # Extract the central values for each channel
        b3lyp_ene = (
            0.72 * x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.81 * x[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.08 * x[:, [2], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.19 * x[:, [3], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        )
        # b3lyp_ene = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        # Reshape the input for the linear layers
        x = x.reshape(-1, self.d_model)

        for i, layer in enumerate(self.layers):
            if IF_SKIP_CONNECTION:
                skip = x
            if i < len(self.layers) - 1:
                x = self.norm[i](x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.actv_fn(x)
                x = self.dropout(x)
            if IF_SKIP_CONNECTION:
                if 1 < i < len(self.layers) - 1:
                    x = x + skip

        # Apply the weighted sum of the central values
        x = x * b3lyp_ene

        return x
