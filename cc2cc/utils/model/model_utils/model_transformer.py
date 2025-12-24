import torch
from torch import nn


class Attention(nn.Module):
    """
    Attention module.
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        self.d_model = kwargs.get("d_model")
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.drop_rate = kwargs.get("drop_rate", 0)

        self.sqrt_d = self.d_model**0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dense_q = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.dense_k = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.dense_v = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)
        self.dense_out = nn.Linear(self.d_model, self.d_model, bias=self.qkv_bias)

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        Time complexity: O(d_model * seq_len^2)
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        q = self.dense_q(inputs)
        k = self.dense_k(inputs)
        v = self.dense_v(inputs)
        # SHAPE qkv = (batch, seq_len, d_model)
        qk = torch.einsum("bqd,bkd->bqk", q, k)
        # SHAPE qk = (batch, seq_len, seq_len)
        attn = self.softmax(qk / self.sqrt_d)
        attn = self.dropout1(attn)
        # SHAPE attn = (batch, seq_len, seq_len)
        qkv = torch.einsum("bqk,bkd->bqd", attn, v)
        # SHAPE qkv = (batch, seq_len, d_model)
        results = self.dense_out(qkv)
        results = self.dropout2(results)
        # SHAPE results = (batch, seq_len, d_model)
        return results


class FFN(nn.Module):
    """
    Feed Forward Network module.
    """

    def __init__(self, **kwargs):
        super(FFN, self).__init__()
        d_model = kwargs.get("d_model")
        mlp_ratio = kwargs.get("mlp_ratio", 1)
        drop_rate = kwargs.get("drop_rate", 0)
        atte_actv = kwargs.get("atte_actv")
        ffn_bias = kwargs.get("ffn_bias", True)

        self.dense1 = nn.Linear(d_model, d_model * mlp_ratio, bias=ffn_bias)
        self.dense2 = nn.Linear(d_model * mlp_ratio, d_model, bias=ffn_bias)

        if atte_actv == "relu":
            self.actv_fn = nn.ReLU()
        elif atte_actv == "gelu":
            self.actv_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {atte_actv}")

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(self, inputs):
        """
        Standard forward function, required for all nn.Module classes
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        results = self.dense1(inputs)
        results = self.actv_fn(results)
        results = self.dropout1(results)
        results = self.dense2(results)
        results = self.dropout2(results)
        # SHAPE results = (batch, seq_len, d_model)
        return results


class Block(nn.Module):
    """
    A block with attention and mlp.
    """

    def __init__(self, **kwargs):
        super(Block, self).__init__()
        d_model = kwargs.get("d_model")
        atte_normal = kwargs.get("atte_normal")

        self.atten = Attention(**kwargs)
        self.ffn = FFN(**kwargs)

        if atte_normal == "layer":
            self.layernorm1 = nn.LayerNorm(d_model)
            self.layernorm2 = nn.LayerNorm(d_model)
        elif atte_normal == "rms":
            self.layernorm1 = nn.RMSNorm(d_model)
            self.layernorm2 = nn.RMSNorm(d_model)
        else:
            self.layernorm1 = nn.Identity()
            self.layernorm2 = nn.Identity()

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # SHAPE inputs = (batch, seq_len, d_model)
        x = self.atten(self.layernorm1(x)) + x
        # SHAPE results = (batch, seq_len, d_model)
        x = self.ffn(self.layernorm2(x)) + x
        # SHAPE results = (batch, seq_len, d_model)
        return x


class Transformer(nn.Module):
    """
    Transformer module.
    """

    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        d_model = kwargs.get("d_model")
        num_layer = kwargs.get("num_layer")
        atte_actv = kwargs.get("atte_actv")

        self.layer_blocks = nn.ModuleList([Block(**kwargs) for _ in range(num_layer)])

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # do attention only when the feature shape is small enough
        # SHAPE x = (batch, seq_len, d_model)
        for layer in self.layer_blocks:
            x = layer(x)

        return x
