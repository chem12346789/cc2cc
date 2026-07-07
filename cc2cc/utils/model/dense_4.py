import torch

from cc2cc.utils.env_var import EDGE_SIZE
from cc2cc.utils.model.model_utils import DenseNet


class Model(torch.nn.Module):
    """Transformer/e3nn mixed model."""

    def __init__(self):
        super().__init__()

        self.cube_type = "cube"
        self.cube_size = EDGE_SIZE**3
        self.cube_middle = (self.cube_size - 1) // 2
        self.input_level = 4
        self.before_weight = True
        self.lmax = 2
        self.out_l = 0

        self.densenet_center = DenseNet(
            d_model=self.input_level,
            mlp=4,
            depth=2,
            dense_bias=False,
            if_skip_connection_dense=0,
            drop_rate=0,
            dense_actv="gelu",
        )

    def forward(self, x):
        x_center = x[:, :, self.cube_middle]
        center_values = x_center.reshape(-1, self.input_level)
        x_center = self.densenet_center(center_values)

        return x_center
