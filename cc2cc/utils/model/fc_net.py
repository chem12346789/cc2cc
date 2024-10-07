import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(
        self,
        input_size=4,
        hidden_size=100,
        output_size=1,
        residual=1,
        num_layers=4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.residual = residual
        self.num_layers = num_layers

        sizes = [input_size] + [self.hidden_size] * self.num_layers + [output_size]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )
        # self.normal = nn.ModuleList(
        #     [
        #         nn.BatchNorm1d(num_features=output_size, track_running_stats=False)
        #         for input_size, output_size in zip(sizes, sizes[1:])
        #     ]
        # )
        self.actv_fn = nn.ReLU()

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        if self.residual == 2:
            res_tmp = torch.zeros(self.hidden_size, device=x.device)
            num_res = 0
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                # tmp = self.normal[i](x)
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
