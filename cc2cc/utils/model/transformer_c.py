"""
Generate list of model.
"""

import torch
from torch import nn

ANG = 302
RAD = 75

D_MODEL = 1
SEQ_LEN = 4
DEPTH_TRANSFORMER = 5
QKV_BIAS = False
DROP_RATE_TRANSFORMER = 0
NUM_HEADS = 1

MLP_DENSE = 128
DEPTH_DENSE = 7
IF_SKIP_CONNECTION_DENSE = 1
DROP_RATE_DENSE = 0


# ATTE_ACTV = "relu"
# ATTE_NORMAL = "layer"

ATTE_ACTV = "gelu"
ATTE_NORMAL = "rms"

# DENSE_ACTV = "relu"
# DENSE_NORMAL = "layer"

DENSE_ACTV = "gelu"
DENSE_NORMAL = "rms"


class Attention(nn.Module):
    """
    Attention module.
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.num_heads = kwargs.get("num_heads", NUM_HEADS)
        self.qkv_bias = kwargs.get("qkv_bias", QKV_BIAS)
        self.sqrt_d = self.d_model**0.5
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE_TRANSFORMER)

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
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.mlp_ratio = kwargs.get("mlp_ratio", 1)
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE_TRANSFORMER)

        self.dense1 = nn.Linear(self.d_model, self.d_model * self.mlp_ratio)
        self.dense2 = nn.Linear(self.d_model * self.mlp_ratio, self.d_model)
        if ATTE_ACTV == "relu":
            self.actv_fn = nn.ReLU()
        elif ATTE_ACTV == "gelu":
            self.actv_fn = nn.GELU()
        self.atten = Attention(**kwargs)

        self.dropout0 = nn.Dropout(self.drop_rate)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)

        if ATTE_NORMAL == "layer":
            self.layernorm1 = nn.LayerNorm(self.d_model)
            self.layernorm2 = nn.LayerNorm(self.d_model)
        elif ATTE_NORMAL == "rms":
            self.layernorm1 = nn.RMSNorm(self.d_model)
            self.layernorm2 = nn.RMSNorm(self.d_model)

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
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.seq_len = kwargs.get("seq_len", SEQ_LEN)
        self.num_layer = kwargs.get("num_layer", DEPTH_TRANSFORMER)
        self.qkv_bias = kwargs.get("qkv_bias", QKV_BIAS)
        self.num_heads = kwargs.get("num_heads", NUM_HEADS)
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE_TRANSFORMER)

        if ATTE_ACTV == "relu":
            self.actv_fn = nn.ReLU()
        elif ATTE_ACTV == "gelu":
            self.actv_fn = nn.GELU()

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
        results = self.actv_fn(results)
        results = self.dropout1(results)
        # SHAPE results = (batch, seq_len, d_model)

        # do attention only when the feature shape is small enough
        for i in range(self.num_layer):
            results = self.layer_blocks[i](results)
        # SHAPE results = (batch, seq_len, d_model)

        results = self.dense2(results)
        results = self.actv_fn(results)
        results = self.head(results)
        # SHAPE results = (batch, seq_len, d_model)
        return results


class DenseNet(nn.Module):
    """
    DenseNet module.
    """

    def __init__(self, **kwargs):
        super(DenseNet, self).__init__()
        self.d_model = kwargs.get("seq_len", SEQ_LEN * D_MODEL)
        self.mlp = kwargs.get("mlp", MLP_DENSE)
        self.depth = kwargs.get("depth_dense", DEPTH_DENSE)
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE_DENSE)

        self.sizes = [self.d_model] + [self.mlp] * (self.depth - 1) + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(self.sizes, self.sizes[1:])
            ]
        )

        if DENSE_ACTV == "relu":
            self.actv_fn = nn.ReLU()
        elif DENSE_ACTV == "gelu":
            self.actv_fn = nn.GELU()

        if DENSE_NORMAL == "layer":
            self.norm = nn.ModuleList(
                [nn.LayerNorm(i_size) for i_size in self.sizes[:-2]]
            )
        elif DENSE_NORMAL == "rms":
            self.norm = nn.ModuleList(
                [nn.RMSNorm(i_size) for i_size in self.sizes[:-2]]
            )

        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        for i, layer in enumerate(self.layers):
            if IF_SKIP_CONNECTION_DENSE:
                skip = x
            if i < len(self.layers) - 1:
                x = self.norm[i](x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.actv_fn(x)
                x = self.dropout(x)
            if IF_SKIP_CONNECTION_DENSE:
                if self.sizes[i] == self.sizes[i + 1]:
                    x = x + skip
        return x


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.model_type = "center_4"

        self.predictor = Extractor(**kwargs)
        self.densenet = DenseNet(**kwargs)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # Extract the central values for each channel
        b3lyp_ene = (
            0.08 * x[:, [0]] + 0.19 * x[:, [1]] + 0.72 * x[:, [2]] + 0.81 * x[:, [3]]
        )
        # b3lyp_ene = x[:, [0]]

        # SHAPE x = (batch, 4)
        x = x.reshape(-1, 4, 1)
        # SHAPE x = (N_ATOM * ANG, SEQ_LEN, D_MODEL)
        x = self.predictor(x)
        # SHAPE shape = (N_ATOM * ANG, SEQ_LEN, D_MODEL)

        # SHAPE x = (batch, 4, 1)
        x = x.reshape(-1, 4 * 1)
        # SHAPE x = (batch, 4 * 1)
        x = self.densenet(x)
        # SHAPE x = (batch, 1)
        x = x * b3lyp_ene
        return x
