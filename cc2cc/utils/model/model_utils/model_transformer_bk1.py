import torch
from torch import nn


class Attention(nn.Module):
    """
    Attention module.
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        d_model = kwargs.get("d_model")
        qkv_bias = kwargs.get("qkv_bias", False)
        bias = kwargs.get("bias", True)

        self.sqrt_d = d_model**0.5
        self.dense_in_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.dense_in_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.dense_in_v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.dense_out = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        Time complexity: O(d_model * seq_len^2)
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        query = self.dense_in_q(inputs)
        key = self.dense_in_k(inputs)
        value = self.dense_in_v(inputs)
        # (batch, seq_len, d_model) @ (batch, seq_len, d_model).T -> (batch, seq_len, seq_len)
        scores = torch.einsum("bqd,bkd->bqk", query, key) / self.sqrt_d
        scores = torch.nn.functional.softmax(scores, dim=-1)
        # (batch, seq_len, seq_len) @ (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        results = torch.einsum("bqk,bkd->bqd", scores, value)
        results = self.dense_out(results)
        return results


class FFN(nn.Module):
    """
    Feed Forward Network module.
    """

    def __init__(self, **kwargs):
        super(FFN, self).__init__()
        d_model = kwargs.get("d_model")
        mlp_ratio = kwargs.get("mlp_ratio", 1)
        atte_actv = kwargs.get("atte_actv", "relu")
        bias = kwargs.get("bias", True)

        if atte_actv == "relu":
            self.actv_fn = nn.ReLU()
        elif atte_actv == "gelu":
            self.actv_fn = nn.GELU()
        elif atte_actv == "mish":
            self.actv_fn = nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {atte_actv}")

        self.dense1 = nn.Linear(d_model, d_model * mlp_ratio, bias=bias)
        self.dense2 = nn.Linear(d_model * mlp_ratio, d_model, bias=bias)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        x = self.dense1(x)
        x = self.actv_fn(x)
        x = self.dense2(x)
        # SHAPE results = (batch, seq_len, d_model)
        return x


class Block(nn.Module):
    """
    A block with attention and mlp.
    """

    def __init__(self, **kwargs):
        super(Block, self).__init__()
        d_model = kwargs.get("d_model")
        atte_normal = kwargs.get("atte_normal")

        if atte_normal == "layer":
            self.attn_norm = nn.LayerNorm(d_model)
            self.ffn_norm = nn.LayerNorm(d_model)
        elif atte_normal == "rms":
            self.attn_norm = nn.RMSNorm(d_model)
            self.ffn_norm = nn.RMSNorm(d_model)
        else:
            self.attn_norm = nn.Identity()
            self.ffn_norm = nn.Identity()

        self.attention = Attention(**kwargs)
        self.ffn = FFN(**kwargs)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        x = self.attention(self.attn_norm(x)) + x
        # SHAPE attention = (batch, seq_len, d_model)

        x = self.ffn(self.ffn_norm(x)) + x
        # SHAPE ffn = (batch, seq_len, d_model)
        return x


class Transformer(nn.Module):
    """
    Transformer module.
    """

    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        num_layer = kwargs.get("num_layer")
        bias = kwargs.get("bias", True)
        d_model = kwargs.get("d_model")

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(Block(**kwargs))
        self.head = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # do attention only when the feature shape is small enough
        for layer in self.layers:
            x = layer(x)
        # SHAPE x = (batch, seq_len, d_model)

        x = self.head(x)
        # SHAPE x = (batch, seq_len, d_model)
        return x
