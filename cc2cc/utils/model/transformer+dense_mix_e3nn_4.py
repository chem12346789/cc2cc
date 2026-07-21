import torch

from cc2cc.utils.env_var import EDGE_SIZE
from cc2cc.utils.model.model_utils import DenseNet, Transformer, E3nn


class Model(torch.nn.Module):
    """Transformer/e3nn mixed model."""

    def __init__(self):
        super().__init__()

        self.cube_type = "cube"
        self.cube_size = EDGE_SIZE**3
        self.cube_middle = (self.cube_size - 1) // 2
        self.input_level = 4
        self.before_weight = False
        self.lmax = 2
        self.out_l = 0
        self.flat_size = self.input_level * self.cube_size

        e3nn_args = (
            self.cube_type,
            self.cube_size,
            self.input_level,
            self.lmax,
            self.out_l,
        )
        for i in range(1, self.input_level + 1):
            setattr(self, f"conv{i}", E3nn(*e3nn_args))

        self.predictor = Transformer(
            d_model=self.cube_size,
            seq_len=self.input_level,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=self.flat_size,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            dense_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=self.input_level,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=0,
            drop_rate=0,
            dense_actv="gelu",
        )

        self.mixing_weight = torch.nn.Linear(self.flat_size, self.input_level + 2)

    def forward(self, x):
        x_center = x[:, :, self.cube_middle]

        x_in = x.permute(0, 2, 1).contiguous()
        x_cube = torch.cat(
            tuple(
                torch.vmap(getattr(self, f"conv{i}"))(x_in)
                for i in range(1, self.input_level + 1)
            ),
            dim=-2,
        )

        weight_out = torch.softmax(
            self.mixing_weight(x_cube.reshape(-1, self.flat_size)), dim=-1
        )

        x_cube = self.densenet(self.predictor(x_cube).reshape(-1, self.flat_size))

        center_values = x_center.reshape(-1, self.input_level)
        x_center = self.densenet_center(center_values)

        expert_outputs = torch.cat((x_cube, x_center, center_values), dim=-1)
        return (weight_out * expert_outputs).sum(dim=-1, keepdim=True)
