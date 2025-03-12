"""
An 3d cnn model
"""

import importlib.resources
import os
from pathlib import Path

import torch.nn as nn

from cc2cc.utils.env_var import CUBE_MIDDLE

D_MODEL = 108
DENSE_DEPTH = 3


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.depth = kwargs.get("depth", DENSE_DEPTH) - 1

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

        sizes = [self.d_model] + [self.d_model] * self.depth + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )
        self.actv_fn = nn.ReLU()
        self.norm = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.depth)]
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        t = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        x = x.reshape(-1, self.d_model)

        for i, layer in enumerate(self.layers):
            # skip = x
            if i < len(self.layers) - 1:
                x = self.norm[i](x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.actv_fn(x)
        x = x * t
        return x
