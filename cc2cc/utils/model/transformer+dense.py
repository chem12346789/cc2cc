"""
Generate list of model.
"""

from torch import nn

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE
from cc2cc.utils.model.model_utils import Transformer, DenseNet


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "cube"

        # self.predictor = Transformer(
        #     d_model=CUBE_SIZE**3,
        #     seq_len=4,
        #     num_layer=3,
        #     qkv_bias=False,
        #     ffn_bias=False,
        #     mlp_ratio=2,
        #     atte_actv="gelu",
        # )

        self.densenet = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=2,
            dense_bias=False,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )

        # self.densenet_center = DenseNet(
        #     d_model=4,
        #     mlp=108,
        #     depth=3,
        #     if_skip_connection_dense=1,
        #     drop_rate=0,
        #     dense_actv="gelu",
        # )

        # self.mixing_weight = nn.Linear(4 * CUBE_SIZE**3, 2)
        # self.weight_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # x = x.reshape(-1, 4, CUBE_SIZE**3)
        # x = self.predictor(x)
        x = x.reshape(-1, 4 * CUBE_SIZE**3)
        x = self.densenet(x)
        return x
