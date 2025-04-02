"""
An 3d cnn model
"""

import importlib.resources
import os
from pathlib import Path

import torch.nn as nn

from cc2cc.utils.env_var import CUBE_MIDDLE

D_MODEL = 108
MLP = 1
DENSE_DEPTH = 3
IF_SKIP_CONNECTION = 1

DROP_RATE = 0


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.mlp = kwargs.get("mlp", MLP)
        self.depth = kwargs.get("depth", DENSE_DEPTH) - 1
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE)

        print("#INFO: **** detail of model ****")
        print(f"#INFO: **** d_model is {self.d_model} ****")
        print(f"#INFO: **** mlp is {self.mlp} ****")
        print(f"#INFO: **** depth is {self.depth} ****")
        print(f"#INFO: **** if_skip_connection is {IF_SKIP_CONNECTION} ****")

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

        self.sizes = [self.d_model] + [self.d_model * self.mlp] * self.depth + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(self.sizes, self.sizes[1:])
            ]
        )

        self.actv_fn = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_rate)
        self.norm = nn.ModuleList([nn.LayerNorm(i_size) for i_size in self.sizes[:-2]])

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        t = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        x = x.reshape(-1, self.d_model)

        for i, layer in enumerate(self.layers):
            if IF_SKIP_CONNECTION:
                skip = x.clone()
            if i < len(self.layers) - 1:
                x = self.norm[i](x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.actv_fn(x)
                x = self.dropout(x)
            if IF_SKIP_CONNECTION:
                if self.sizes[i] == self.sizes[i + 1]:
                    x = x + skip
        x = x * t
        return x
