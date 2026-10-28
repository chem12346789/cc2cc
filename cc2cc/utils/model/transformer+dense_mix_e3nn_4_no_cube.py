"""Transformer mixture for center-first, irregular quadrature neighborhoods."""

import torch

from cc2cc.utils.model.model_utils.model_dense import DenseNet
from cc2cc.utils.model.model_utils.model_transformer import Transformer
from cc2cc.utils.model.model_utils.model_e3nn_correct_no_cube import E3nn


class Model(torch.nn.Module):
    """Four scalar features per point, relative positions in Bohr, scalar output."""

    def __init__(self):
        super().__init__()
        self.cube_type = "neighbors"
        self.cube_size = 27
        self.cube_middle = 0
        self.input_level = 4
        self.before_weight = False
        self.flat_size = self.input_level * self.cube_size
        self.lmax = 0
        for i in range(1, self.input_level + 1):
            setattr(
                self,
                f"conv{i}",
                E3nn(self.cube_type, self.cube_size, self.input_level, self.lmax),
            )
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

    def forward(self, x, relative_positions):
        if x.ndim != 3 or x.shape[1:] != (4, 27):
            raise ValueError("Expected density features with shape (N, 4, 27).")
        if relative_positions.shape != (x.shape[0], 27, 3):
            raise ValueError("Expected relative positions with shape (N, 27, 3).")
        features = x.permute(0, 2, 1).contiguous()
        encoded = torch.cat(
            tuple(
                getattr(self, f"conv{i}")(features, relative_positions)
                for i in range(1, self.input_level + 1)
            ),
            dim=-2,
        )
        mixing = torch.softmax(self.mixing_weight(encoded.flatten(1)), dim=-1)
        neighborhood = self.densenet(self.predictor(encoded).flatten(1))
        center = x[:, :, 0]
        experts = torch.cat(
            (neighborhood, self.densenet_center(center), center), dim=-1
        )
        return (mixing * experts).sum(dim=-1, keepdim=True)
