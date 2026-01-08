"""
Generate list of model.
"""

import torch
from torch import nn

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE


class Attention(nn.Module):
    """
    Attention module.
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        self.d_model = kwargs.get("d_model")
        self.num_heads = kwargs.get("num_heads")
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.drop_rate = kwargs.get("drop_rate", 0)
        self.sqrt_d = self.d_model**0.5

        self.dense1 = nn.Linear(self.d_model, self.d_model * 3, bias=self.qkv_bias)
        self.dense2 = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        Time complexity: O(d_model * seq_len^2)
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        b, s, _ = inputs.shape
        results = self.dense1(inputs)
        # SHAPE results = (batch, seq_len, 3 * d_model)
        results = torch.reshape(
            results, (b, s, 3, self.num_heads, self.d_model // self.num_heads)
        )
        # SHAPE results = (batch, seq_len, 3, head, d_model // head)
        results = torch.permute(results, (0, 2, 3, 1, 4))
        # SHAPE results = (batch, 3, head, seq_len, d_model // head)
        q, k, v = (
            results[:, 0, ...],
            results[:, 1, ...],
            results[:, 2, ...],
        )
        # SHAPE qkv = (batch, head, seq_len, d_model // head)
        qk = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_d
        # SHAPE qk = (batch, head, seq_len, seq_len)
        attn = torch.softmax(qk, dim=-1)
        attn = self.dropout1(attn)
        # SHAPE attn = (batch, head, seq_len, seq_len)
        qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3))
        # SHAPE qkv = (batch, seq_len, head, d_model // head)
        qkv = torch.reshape(qkv, (b, s, self.d_model))
        # SHAPE qkv = (batch, seq_len, d_model)
        results = self.dense2(qkv)
        results = self.dropout2(results)
        # SHAPE results = (batch, seq_len, d_model)
        return results


class ABlock(nn.Module):
    """
    A block with attention and mlp.
    """

    def __init__(self, **kwargs):
        super(ABlock, self).__init__()
        self.d_model = kwargs.get("d_model")
        self.mlp_ratio = kwargs.get("mlp_ratio", 1)
        self.drop_rate = kwargs.get("drop_rate", 0)
        self.atte_actv = kwargs.get("atte_actv")
        self.atte_normal = kwargs.get("atte_normal")

        self.dense1 = nn.Linear(self.d_model, self.d_model * self.mlp_ratio)
        self.dense2 = nn.Linear(self.d_model * self.mlp_ratio, self.d_model)
        if self.atte_actv == "relu":
            self.actv_fn = nn.ReLU()
        elif self.atte_actv == "gelu":
            self.actv_fn = nn.GELU()
        elif self.atte_actv == "mish":
            self.actv_fn = nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {self.actv_fn}")

        self.atten = Attention(**kwargs)

        self.dropout0 = nn.Dropout(self.drop_rate)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)

        if self.atte_normal == "layer":
            self.layernorm1 = nn.LayerNorm(self.d_model)
            self.layernorm2 = nn.LayerNorm(self.d_model)
        elif self.atte_normal == "rms":
            self.layernorm1 = nn.RMSNorm(self.d_model)
            self.layernorm2 = nn.RMSNorm(self.d_model)
        else:
            self.layernorm1 = nn.Identity()
            self.layernorm2 = nn.Identity()

    def forward(self, x_inp):
        """
        Standard forward function, required for all nn.Module classes
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        x_attn = self.layernorm1(x_inp)
        x_attn = self.atten(x_attn)
        x_attn = self.dropout0(x_attn)
        x_attn = x_attn + x_inp
        # SHAPE results = (batch, seq_len, d_model)

        x_mlp = self.layernorm2(x_attn)
        x_mlp = self.dense1(x_mlp)
        x_mlp = self.actv_fn(x_mlp)
        x_mlp = self.dropout1(x_mlp)
        x_mlp = self.dense2(x_mlp)
        x_mlp = self.dropout2(x_mlp)
        x_mlp = x_mlp + x_attn
        # SHAPE results = (batch, seq_len, d_model)
        return x_mlp


class Extractor(nn.Module):
    """
    Extractor module.
    """

    def __init__(self, **kwargs):
        super(Extractor, self).__init__()
        self.d_model = kwargs.get("d_model")
        self.seq_len = kwargs.get("seq_len")
        self.num_layer = kwargs.get("num_layer")
        self.qkv_bias = kwargs.get("qkv_bias")
        self.num_heads = kwargs.get("num_heads")
        self.drop_rate = kwargs.get("drop_rate")
        self.atte_actv = kwargs.get("atte_actv")
        self.atte_normal = kwargs.get("atte_normal")

        if self.atte_actv == "relu":
            self.actv_fn1 = nn.ReLU()
            self.actv_fn2 = nn.ReLU()
        elif self.atte_actv == "gelu":
            self.actv_fn1 = nn.GELU()
            self.actv_fn2 = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {self.actv_fn}")

        self.dense1 = nn.Linear(self.d_model, self.d_model)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.layer_blocks = nn.ModuleList(
            [ABlock(**kwargs) for _ in range(self.num_layer)]
        )
        self.dense2 = nn.Linear(self.d_model, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_model)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        results = self.dense1(inputs)
        results = self.actv_fn1(results)
        results = self.dropout1(results)
        # SHAPE results = (batch, seq_len, d_model)

        # do attention only when the feature shape is small enough
        for i in range(self.num_layer):
            results = self.layer_blocks[i](results)
        # SHAPE results = (batch, seq_len, d_model)

        results = self.dense2(results)
        results = self.actv_fn2(results)
        results = self.head(results)
        # SHAPE results = (batch, seq_len, d_model)
        return results


class DenseNet(nn.Module):
    """
    DenseNet module.
    """

    def __init__(self, **kwargs):
        super(DenseNet, self).__init__()
        self.d_model = kwargs.get("d_model")
        self.mlp = kwargs.get("mlp")
        self.depth = kwargs.get("depth")
        self.drop_rate = kwargs.get("drop_rate", 0.0)
        self.dense_bias = kwargs.get("dense_bias", True)
        self.dense_actv = kwargs.get("dense_actv", "gelu")
        self.dense_normal = kwargs.get("dense_normal", "")
        self.if_skip_connection_dense = kwargs.get("if_skip_connection_dense", True)

        self.sizes = [self.d_model] + [self.mlp] * (self.depth - 1) + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size, bias=self.dense_bias)
                for input_size, output_size in zip(self.sizes, self.sizes[1:])
            ]
        )

        if self.dense_actv == "relu":
            self.actv_fn = nn.ReLU()
        elif self.dense_actv == "gelu":
            self.actv_fn = nn.GELU()
        elif self.dense_actv == "mish":
            self.actv_fn = nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {self.actv_fn}")

        if self.dense_normal == "layer":
            self.norm = nn.ModuleList(
                [nn.LayerNorm(i_size) for i_size in self.sizes[:-2]]
            )
        elif self.dense_normal == "rms":
            self.norm = nn.ModuleList(
                [nn.RMSNorm(i_size) for i_size in self.sizes[:-2]]
            )
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in self.sizes[:-2]])

        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        for i, layer in enumerate(self.layers):
            if self.if_skip_connection_dense:
                skip = x
            if i < len(self.layers) - 1:
                x = self.norm[i](x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.actv_fn(x)
                x = self.dropout(x)
            if self.if_skip_connection_dense:
                if self.sizes[i] == self.sizes[i + 1]:
                    x = x + skip
        return x


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self):
        super().__init__()

        self.model_type = "cube"
        self.input_level = 4

        self.predictor = Extractor(
            d_model=CUBE_SIZE**3,
            seq_len=4,
            num_layer=3,
            qkv_bias=False,
            num_heads=1,
            mlp_ratio=1,
            drop_rate=0,
            atte_actv="gelu",
        )

        self.densenet = DenseNet(
            d_model=4 * CUBE_SIZE**3,
            mlp=108,
            depth=9,
            drop_rate=0,
            if_skip_connection_dense=1,
            dense_actv="gelu",
        )

        self.predictor_center = Extractor(
            d_model=1,
            seq_len=4,
            num_layer=7,
            qkv_bias=False,
            num_heads=1,
            mlp_ratio=1,
            drop_rate=0,
            atte_actv="gelu",
        )

        self.densenet_center = DenseNet(
            d_model=4,
            mlp=128,
            depth=9,
            if_skip_connection_dense=1,
            drop_rate=0,
            dense_actv="gelu",
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # Extract the central values for each channel
        b3lyp_ene = (
            0.08 * x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.19 * x[:, [1], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.72 * x[:, [2], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
            + 0.81 * x[:, [3], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        )
        # b3lyp_ene = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        x_center = x[:, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]

        # SHAPE x = (batch, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        x = x.reshape(-1, 4, CUBE_SIZE**3)
        # SHAPE x = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x = self.predictor(x)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)

        # SHAPE x = (batch, 4, CUBE_SIZE**3)
        x = x.reshape(-1, 4 * CUBE_SIZE**3)
        # SHAPE x = (batch, 4 * CUBE_SIZE**3)
        x = self.densenet(x)
        # SHAPE x = (batch, 1)

        # SHAPE x_center = (batch, 4)
        x_center = x_center.reshape(-1, 4, 1)
        # SHAPE x_center = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)
        x_center = self.predictor_center(x_center)
        # SHAPE shape = (N_ATOM * NGRIDS, SEQ_LEN, D_MODEL)

        # SHAPE x_center = (batch, 4, 1)
        x_center = x_center.reshape(-1, 4 * 1)
        # SHAPE x_center = (batch, 4 * 1)
        x_center = self.densenet_center(x_center)
        # SHAPE x_center = (batch, 1)

        return b3lyp_ene * (x + x_center)
