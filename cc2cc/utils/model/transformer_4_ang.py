"""
Generate list of model.
"""

import importlib.resources
import os
from pathlib import Path

import torch
from torch import nn

from cc2cc.utils.env_var import CUBE_MIDDLE

ANG = 302
RAD = 75

D_MODEL = RAD
SEQ_LEN = 4
NUM_LAYER_TRANSFORMER = 7
NUM_HEADS = 1

L_DENSE = 108
NUM_LAYER_DENSE = 3

QKV_BIAS = False
DROP_RATE = 0

# ATTE_ACTV = "gelu"
ATTE_ACTV = "relu"

DENSE_ACTV = "relu"


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

        self.dense1 = nn.Linear(self.d_model, self.d_model * 3, bias=self.qkv_bias)
        self.dense2 = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        Time complexity: O(d_model * seq_len^2)
        """
        # inputs.shape = (batch, seq_len, d_model)
        results = self.dense1(inputs)
        # results.shape = (batch, seq_len, 3 * d_model)
        b, s, _ = results.shape
        results = torch.reshape(
            results, (b, s, 3, self.num_heads, self.d_model // self.num_heads)
        )
        # results.shape = (batch, seq_len, 3, head, d_model // head)
        results = torch.permute(results, (0, 2, 3, 1, 4))
        # results.shape = (batch, 3, head, seq_len, d_model // head)
        q, k, v = (
            results[:, 0, ...],
            results[:, 1, ...],
            results[:, 2, ...],
        )
        # shape = (batch, head, seq_len, d_model // head)
        qk = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_d
        # qk.shape = (batch, head, seq_len, seq_len)
        attn = torch.softmax(qk, dim=-1)
        # attn.shape = (batch, head, seq_len, seq_len)
        qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3))
        # qkv.shape = (batch, seq_len, head, d_model // head)
        qkv = torch.reshape(qkv, (b, s, self.d_model))
        # qkv.shape = (batch, seq_len, d_model)
        results = self.dense2(qkv)
        # results.shape = (batch, seq_len, d_model)
        return results


class ABlock(nn.Module):
    """
    A block with attention and mlp.
    """

    def __init__(self, **kwargs):
        super(ABlock, self).__init__()
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.mlp_ratio = kwargs.get("mlp_ratio", 1)
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE)

        self.dense1 = nn.Linear(self.d_model, self.d_model * self.mlp_ratio)
        self.dense2 = nn.Linear(self.d_model * self.mlp_ratio, self.d_model)
        if ATTE_ACTV == "relu":
            self.actv_fn = nn.ReLU()
        elif ATTE_ACTV == "gelu":
            self.actv_fn = nn.GELU()
        self.atten = Attention(**kwargs)

        self.dropout_atten = nn.Dropout(self.drop_rate)
        self.dropout_mlp = nn.Dropout(self.drop_rate)
        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)

    def forward(self, x_inp):
        """
        Standard forward function, required for all nn.Module classes
        """
        # SHAPE: inputs = (batch, seq_len, d_model)
        x_attn = self.layernorm1(x_inp)
        x_attn = self.atten(x_attn)
        x_attn = self.dropout_atten(x_attn)
        x_attn += x_inp
        # SHAPE: results = (batch, seq_len, d_model)

        x_mlp = self.layernorm2(x_attn)
        x_mlp = self.dense1(x_mlp)
        x_mlp = self.actv_fn(x_mlp)
        x_mlp = self.dropout_mlp(x_mlp)
        x_mlp = self.dense2(x_mlp)
        x_mlp += x_attn
        # SHAPE: results = (batch, seq_len, d_model)
        return x_mlp


class Extractor(nn.Module):
    """
    Extractor module.
    """

    def __init__(self, **kwargs):
        super(Extractor, self).__init__()
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.seq_len = kwargs.get("seq_len", SEQ_LEN)
        self.num_layer = kwargs.get("num_layer", NUM_LAYER_TRANSFORMER)
        self.qkv_bias = kwargs.get("qkv_bias", QKV_BIAS)
        self.num_heads = kwargs.get("num_heads", NUM_HEADS)
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE)

        if ATTE_ACTV == "relu":
            self.actv_fn = nn.ReLU()
        elif ATTE_ACTV == "gelu":
            self.actv_fn = nn.GELU()
        self.dense1 = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(self.drop_rate)
        self.layer_blocks = nn.ModuleList(
            [ABlock(**kwargs) for _ in range(self.num_layer)]
        )
        self.dense2 = nn.Linear(self.d_model, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_model)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        """
        # inputs.shape = (batch, seq_len, d_model)
        results = self.dense1(inputs)
        results = self.actv_fn(results)
        # results.shape = (batch, seq_len, d_model)

        # do attention only when the feature shape is small enough
        for i in range(self.num_layer):
            results = self.layer_blocks[i](results)
        # results.shape = (batch, seq_len, d_model)

        results = self.dense2(results)
        results = self.actv_fn(results)
        results = self.dropout(results)
        results = self.head(results)
        # results.shape = (batch, seq_len, d_model)
        return results


class DenseNet(nn.Module):
    """
    DenseNet module.
    """

    def __init__(self, **kwargs):
        super(DenseNet, self).__init__()
        self.d_model = kwargs.get("seq_len", L_DENSE)
        self.num_layer_dense = kwargs.get("num_layer_dense", NUM_LAYER_DENSE) - 1
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE)

        sizes = [4] + [self.d_model] * self.num_layer_dense + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )

        self.dropout = nn.Dropout(self.drop_rate)

        if DENSE_ACTV == "relu":
            self.actv_fn = nn.ReLU()
        elif DENSE_ACTV == "gelu":
            self.actv_fn = nn.GELU()

        self.norm = nn.ModuleList(
            [nn.LayerNorm(input_size) for input_size in sizes[:-1]]
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.norm[i](x)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.actv_fn(x)
                x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self, **kwargs):
        super().__init__()

        print("#INFO: **** detail of model ****")
        print(f"#INFO: **** ANG is {ANG} ****")
        print(f"#INFO: **** RAD is {RAD} ****")
        print(f"#INFO: **** D_MODEL is {D_MODEL} ****")
        print(f"#INFO: **** SEQ_LEN is {SEQ_LEN} ****")
        print(f"#INFO: **** DEPTH is {NUM_LAYER_TRANSFORMER} ****")
        print(f"#INFO: **** NUM_LAYER_DENSE is {NUM_LAYER_DENSE} ****")
        print(f"#INFO: **** QKV_BIAS is {QKV_BIAS} ****")
        print(f"#INFO: **** NUM_HEADS is {NUM_HEADS} ****")
        print(f"#INFO: **** DROP_RATE is {DROP_RATE} ****")

        # print all contain in this file, for debugging and logging
        with importlib.resources.files("cc2cc").joinpath(
            "utils/model"
        ) as resource_path:
            file_path = Path(os.fspath(resource_path)) / "transformer.py"
            with open(file_path, "r", encoding="utf-8") as finput:
                print(f"#INFO: **** input file is {file_path} ****\n")
                print(finput.read())
                print("#INFO: ****************** input file end ******************\n")
                print("\n")

        self.predictor = Extractor(**kwargs)
        self.densenet = DenseNet(**kwargs)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        t = x[:, [0]]

        if D_MODEL == RAD:
            # SHAPE: x = (batch, 4)
            x = x.reshape(-1, ANG, RAD, 4)
            # SHAPE: x = (N_ATOM, ANG, RAD, 4)
            x = x.reshape(-1, RAD, 4)
            # SHAPE: x = (N_ATOM * ANG, RAD, 4)
            x = torch.permute(x, (0, 2, 1))
            # SHAPE: x = (N_ATOM * ANG, 4, RAD)

        # x.shape = (N_ATOM * ANG, SEQ_LEN, D_MODEL)
        x = self.predictor(x)
        # x.shape = (N_ATOM * ANG, SEQ_LEN, D_MODEL)

        # x.shape = (batch, 4)
        x = x.reshape(-1, 4)
        # x.shape = (batch, 4)
        x = self.densenet(x)
        # x.shape = (batch, 1)
        x = x * t
        return x
