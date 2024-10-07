import torch
import torch.nn as nn

from cc2cc.utils.env_var import CUBE_USE


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self):
        super().__init__()

        self.input_size = 4 * CUBE_USE * CUBE_USE * CUBE_USE
        self.hidden_size = 4 * CUBE_USE * CUBE_USE * CUBE_USE
        self.output_size = 1
        self.residual = 1  # skip connection
        self.num_layers = 4

        sizes = (
            [self.input_size]
            + [self.hidden_size] * self.num_layers
            + [self.output_size]
        )

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )
        self.normal = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features=output_size, track_running_stats=False)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )
        self.actv_fn = nn.ReLU()

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x = x.view(-1, 4 * CUBE_USE * CUBE_USE * CUBE_USE)

        # input size = torch.Size([1, 4, CUBE_USE, CUBE_USE, CUBE_USE])
        # fallten the 3d cube to 1d
        # 4 * CUBE_USE * CUBE_USE * CUBE_USE
        if self.residual == 2:
            res_tmp = torch.zeros(self.hidden_size, device=x.device)
            num_res = 0
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                tmp = self.normal[i](x)
                tmp = self.actv_fn(tmp)
            if layer.in_features == layer.out_features:
                if self.residual == 2:
                    num_res = num_res + 1
                    res_tmp = res_tmp + tmp
                    x = x + res_tmp / num_res
                if self.residual == 1:
                    x = x + tmp
                else:
                    x = tmp
            else:
                x = tmp
        return x
