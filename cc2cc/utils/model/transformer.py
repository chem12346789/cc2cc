"""
Generate list of model.
"""

import importlib.resources
import os
from pathlib import Path

import torch
from torch import nn

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE

ANG = 302
RAD = 75

D_MODEL = RAD
SEQ_LEN = 4 * CUBE_SIZE**3
DEPTH = 5
DENSE_DEPTH = 2

QKV_BIAS = False
NUM_HEADS = 1
DROP_RATE = 0


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
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE)

        self.dense1 = nn.Linear(self.d_model, self.d_model * 3, bias=self.qkv_bias)
        self.dense2 = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)

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
        qk = torch.matmul(q, torch.transpose(k, 2, 3)) / self.sqrt_d
        # qk.shape = (batch, head, seq_len, seq_len)
        attn = torch.softmax(qk, dim=-1)
        attn = self.dropout1(attn)
        # attn.shape = (batch, head, seq_len, seq_len)
        qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3))
        # qkv.shape = (batch, seq_len, head, d_model // head)
        qkv = torch.reshape(qkv, (b, s, self.d_model))
        # qkv.shape = (batch, seq_len, d_model)
        results = self.dense2(qkv)
        results = self.dropout2(results)
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
        self.gelu = nn.GELU()
        self.atten = Attention(**kwargs)

        self.dropout0 = nn.Dropout(self.drop_rate)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        """
        # inputs.shape = (batch, seq_len, d_model)
        skip = inputs
        results = self.layernorm1(inputs)
        results = self.atten(results) + skip
        results = self.dropout0(results)
        # results.shape = (batch, seq_len, d_model)

        skip = results
        results = self.layernorm2(results)
        results = self.dense1(results)
        results = self.gelu(results)
        results = self.dropout1(results)
        results += skip
        skip = results
        results = self.dense2(results)
        results = self.dropout2(results)
        results += skip
        # results.shape = (batch, seq_len, d_model)
        return results


class Extractor(nn.Module):
    """
    Extractor module.
    """

    def __init__(self, **kwargs):
        super(Extractor, self).__init__()
        self.d_model = kwargs.get("d_model", D_MODEL)
        self.seq_len = kwargs.get("seq_len", SEQ_LEN)
        self.depth = kwargs.get("depth", DEPTH)
        self.qkv_bias = kwargs.get("qkv_bias", QKV_BIAS)
        self.num_heads = kwargs.get("num_heads", NUM_HEADS)
        self.drop_rate = kwargs.get("drop_rate", DROP_RATE)

        self.gelu = nn.GELU()
        self.dense1 = nn.Linear(self.d_model, self.d_model)

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.layer_blocks = nn.ModuleList([ABlock(**kwargs) for _ in range(self.depth)])
        self.dense2 = nn.Linear(self.d_model, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_model)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        """
        # batch = inputs.shape[0]
        # inputs.shape = (batch, seq_len, d_model)
        results = self.dense1(inputs)
        results = self.gelu(results)
        results = self.dropout1(results)
        # results.shape = (batch, seq_len, d_model)
        # do attention only when the feature shape is small enough
        for i in range(self.depth):
            results = self.layer_blocks[i](results)
        # results.shape = (batch, seq_len, d_model)
        results = self.dense2(results)
        results = self.gelu(results)
        results = self.head(results)
        # results.shape = (batch, seq_len, d_model)
        return results


class DenseNet(nn.Module):
    """
    DenseNet module.
    """

    def __init__(self, **kwargs):
        super(DenseNet, self).__init__()
        self.d_model = kwargs.get("seq_len", SEQ_LEN)
        self.depth = kwargs.get("depth", DENSE_DEPTH) - 1

        sizes = [self.d_model] + [self.d_model] * self.depth + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )
        self.actv_fn = nn.ReLU()
        self.norm = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.depth)]
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
        print(f"#INFO: **** DEPTH is {DEPTH} ****")
        print(f"#INFO: **** DENSE_DEPTH is {DENSE_DEPTH} ****")
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
                print("\n")

        self.predictor = Extractor(**kwargs)
        self.densenet = DenseNet(**kwargs)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        t = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        # x.shape = (batch, 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        x = torch.permute(
            x.reshape(-1, ANG, RAD, 4 * CUBE_SIZE**3),
            (0, 2, 1, 3),
        )
        # x.shape = (N_ATOM, ANG, RAD, 4 * CUBE_SIZE**3)
        x = torch.permute(x.reshape(-1, RAD, 4 * CUBE_SIZE**3), (0, 2, 1))
        # x.shape = (N_ATOM * ANG, 4 * CUBE_SIZE**3, RAD)

        x = self.predictor(x)
        # x.shape = (N_ATOM * ANG, 4 * CUBE_SIZE**3, RAD)
        x = torch.permute(x, (0, 2, 1)).reshape(-1, 4 * CUBE_SIZE**3)
        # x.shape = (N_ATOM * ANG * RAD, 4 * CUBE_SIZE**3)
        x = self.densenet(x)
        # x.shape = (N_ATOM * ANG * RAD, 1)
        x = x * t
        return x
