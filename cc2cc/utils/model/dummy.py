import torch

from cc2cc.utils.helper.runtime.env_var import EDGE_SIZE


class Model(torch.nn.Module):
    """Minimal model for checkpoint-loading validation."""

    def __init__(self):
        super().__init__()

        self.dummy_scale = torch.nn.Parameter(torch.zeros(()))
        self.cube_type = "cube"
        self.cube_size = EDGE_SIZE**3
        self.cube_middle = (self.cube_size - 1) // 2
        self.input_level = 4
        self.before_weight = True
        self.lmax = 2
        self.out_l = 0

    def forward(self, x):
        x_center = x[:, :, self.cube_middle] * 0

        return x_center * self.dummy_scale
