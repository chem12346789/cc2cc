from importlib import import_module

import torch


BaseModel = import_module(
    "cc2cc.utils.model.transformer+dense_mix_e3nn_4_correct"
).Model


class Model(BaseModel):
    """Four-channel mixed model with learnable center-value scaling."""

    def __init__(self):
        super().__init__()
        self.center_scale = torch.nn.Parameter(
            torch.ones(self.input_level, dtype=torch.float64, device="cpu")
        )
        self.to(device="cpu", dtype=torch.float64)

    def forward(self, x):
        x_center = x[:, :, self.cube_middle]
        x_in = x.permute(0, 2, 1).contiguous()

        x_cube = torch.cat(
            tuple(
                (getattr(self, f"conv{i}"))(x_in)
                for i in range(1, self.input_level + 1)
            ),
            dim=-2,
        )

        weight_out = torch.softmax(
            self.mixing_weight(x_cube.reshape(-1, self.flat_size)), dim=-1
        )

        x_cube = self.densenet(self.predictor(x_cube).reshape(-1, self.flat_size))

        center_values = x_center.reshape(-1, self.input_level) * self.center_scale
        x_center = self.densenet_center(center_values)

        expert_outputs = torch.cat((x_cube, x_center, center_values), dim=-1)
        return (weight_out * expert_outputs).sum(dim=-1, keepdim=True)
