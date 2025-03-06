"""
Generate list of model.
"""

import torch
from torch import nn

from cc2cc.utils.env_var import CUBE_SIZE, CUBE_MIDDLE

D_MODEL = CUBE_SIZE**3
SEQ_LEN = 4
NUM_HEADS = 1
QKV_BIAS = False
DEPTH = 3
DROP_RATE = 0.1


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
        qk = torch.matmul(q, torch.transpose(k, 2, 3)) / self.sqrt_d
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

        self.dense1 = nn.Linear(self.d_model, self.d_model * self.mlp_ratio)
        self.dense2 = nn.Linear(self.d_model * self.mlp_ratio, self.d_model)
        self.gelu = nn.GELU()
        self.atten = Attention(**kwargs)
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
        # results.shape = (batch, seq_len, d_model)
        skip = results
        results = self.layernorm2(results)
        results = self.dense1(results)
        results = self.gelu(results) + skip
        skip = results
        results = self.dense2(results) + skip
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
        self.dense2 = nn.Linear(
            self.seq_len * self.d_model, self.seq_len * self.d_model
        )
        self.head = nn.Linear(self.seq_len * self.d_model, 1, bias=False)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        """
        # batch = inputs.shape[0]
        # # inputs.shape = (bacth, seq_len, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        inputs = inputs.reshape(-1, self.seq_len, self.d_model)
        # inputs.shape = (batch, seq_len, d_model)
        results = self.dense1(inputs)
        results = self.gelu(results)
        results = self.dropout1(results)
        # results.shape = (batch, seq_len, d_model)
        # do attention only when the feature shape is small enough
        for i in range(self.depth):
            results = self.layer_blocks[i](results)
        # results.shape = (batch, seq_len, d_model)
        results = results.reshape(-1, self.seq_len * self.d_model)
        # results.shape = (batch, seq_len * d_model)
        results = self.dense2(results)
        results = self.gelu(results)
        results = self.head(results)
        # results.shape = (batch, 1)
        return results


class Model(nn.Module):
    """
    Transformer module.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.predictor = Extractor(**kwargs)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        t = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        x = self.predictor(x)
        x = x * t
        return x
