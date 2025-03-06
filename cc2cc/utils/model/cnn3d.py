""" 
An 3d cnn model
"""

import importlib.resources
import os
from pathlib import Path

import torch.nn as nn

from cc2cc.utils.env_var import CUBE_MIDDLE


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, **kwargs):
        super().__init__()

        # print all contain in this file, for debugging and logging
        with importlib.resources.files("cc2cc").joinpath(
            "utils/model"
        ) as resource_path:
            file_path = Path(os.fspath(resource_path)) / "cnn3d.py"
            with open(file_path, "r", encoding="utf-8") as finput:
                print(f"#INFO: **** input file is {file_path} ****\n")
                print(finput.read())
                print("#INFO: ****************** input file end ******************\n")
                print("\n")
                print("\n")

        sizes = [108] + [108] * 4 + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )
        self.actv_fn = nn.GELU()
        self.norm = nn.ModuleList(
            [nn.LayerNorm(108) for _ in range(len(self.layers) - 1)]
        )
        self.dropout = nn.ModuleList([nn.Dropout(0.1) for _ in range(len(self.layers))])

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        t = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        x = x.reshape(-1, 108)

        for i, layer in enumerate(self.layers):
            # skip = x
            if i < len(self.layers) - 1:
                x = self.norm[i](x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.actv_fn(x)
            x = self.dropout[i](x)
        x = x * t
        return x
