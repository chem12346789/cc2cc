import torch

from cc2cc.utils.helper.runtime.env_var import CUBE_MIDDLE, EDGE_SIZE

from cc2cc.utils.model.model_utils.model_dense import DenseNet
from cc2cc.utils.model.model_utils.model_transformer import Transformer
from cc2cc.utils.model.model_utils.model_e3nn_correct import E3nn


class Model(torch.nn.Module):
    """Four spatial channels plus a center-only VV10 energy-density channel.

    Input shape is [batch, 5, EDGE_SIZE**3]. The first four channels follow
    transformer+dense_mix_e3nn_4_correct; channel five contains unweighted
    VV10 energy per volume at cube_middle. Its other entries are ignored.
    """

    def __init__(self):
        super().__init__()

        self.cube_type = "cube"
        self.cube_size = EDGE_SIZE**3
        self.cube_middle = CUBE_MIDDLE * (EDGE_SIZE**2 + EDGE_SIZE + 1)
        self.cube_input_level = 4
        self.input_level = 5
        self.requires_vv10_input = True
        self.before_weight = False
        self.lmax = 0
        self.flat_size = self.cube_input_level * self.cube_size

        e3nn_args = {
            "cube_type": self.cube_type,
            "cube_size": self.cube_size,
            "input_level": self.cube_input_level,
            "lmax": self.lmax,
        }
        for i in range(1, self.cube_input_level + 1):
            setattr(self, f"conv{i}", E3nn(**e3nn_args))

        self.predictor = Transformer(
            d_model=self.cube_size,
            seq_len=self.cube_input_level,
            num_layer=5,
            qkv_bias=False,
            ffn_bias=False,
            mlp_ratio=1,
            atte_actv="gelu",
        ).to(device="cpu", dtype=torch.float64)

        self.densenet = DenseNet(
            d_model=self.flat_size,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        ).to(device="cpu", dtype=torch.float64)

        self.densenet_center = DenseNet(
            d_model=self.input_level,
            mlp=108,
            depth=5,
            dense_bias=False,
            if_skip_connection_dense=1,
            drop_rate=0,
            dense_actv="gelu",
        ).to(device="cpu", dtype=torch.float64)

        self.mixing_weight = torch.nn.Linear(
            self.flat_size + 1,
            self.input_level + 2,
            device="cpu",
            dtype=torch.float64,
        )

    def forward(self, x):
        center_values = x[:, :, self.cube_middle]
        vv10_center = center_values[:, self.cube_input_level:]
        x_in = x[:, :self.cube_input_level].permute(0, 2, 1).contiguous()

        x_cube = torch.cat(
            tuple(
                (getattr(self, f"conv{i}"))(x_in)
                for i in range(1, self.cube_input_level + 1)
            ),
            dim=-2,
        )

        mixing_input = torch.cat(
            (x_cube.reshape(-1, self.flat_size), vv10_center), dim=-1
        )
        weight_out = torch.softmax(self.mixing_weight(mixing_input), dim=-1)

        x_cube = self.densenet(self.predictor(x_cube).reshape(-1, self.flat_size))
        x_center = self.densenet_center(center_values)

        expert_outputs = torch.cat((x_cube, x_center, center_values), dim=-1)
        return (weight_out * expert_outputs).sum(dim=-1, keepdim=True)
