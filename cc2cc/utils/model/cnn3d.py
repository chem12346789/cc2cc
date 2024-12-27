# An 3d cnn model
import torch
import torch.nn as nn

from cc2cc.utils.env_var import CUBE_MIDDLE

ESP = torch.finfo(torch.float32).eps


class Attention(nn.Module):
    """
    Attention
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        self.channel = kwargs.get("channel")
        self.seq = kwargs.get("seq")
        self.num_heads = kwargs.get("num_heads")

        self.dense1 = nn.Linear(self.channel, self.channel * 3)
        self.dense2 = nn.Linear(self.channel, self.channel)

    def forward(self, inputs):
        # inputs.shape = (batch, seq, channel)
        results = self.dense1(inputs)
        # results.shape = (batch, seq, 3 * channel)
        results = torch.reshape(
            results, (-1, self.seq, 3, self.num_heads, self.channel // self.num_heads)
        )
        # results.shape = (batch, seq, 3, head, channel // head)
        results = torch.permute(results, (0, 2, 3, 1, 4))
        # results.shape = (batch, 3, head, seq, channel // head)
        q, k, v = (
            results[:, 0, ...],
            results[:, 1, ...],
            results[:, 2, ...],
        )
        # shape = (batch, head, seq, channel // head)
        qk = torch.matmul(q, torch.transpose(k, 2, 3))
        # qk.shape = (batch, head, seq, seq)
        attn = torch.softmax(qk, dim=-1)
        # attn.shape = (batch, head, seq, seq)
        qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3))
        # qkv.shape = (batch, seq, head, channel // head)
        qkv = torch.reshape(qkv, (-1, self.seq, self.channel))
        # qkv.shape = (batch, seq, channel)
        results = self.dense2(qkv)
        # results.shape = (batch, seq, channel)
        return results


class ABlock(nn.Module):
    def __init__(self, **kwargs):
        super(ABlock, self).__init__()
        self.channel = kwargs.get("channel")
        self.mlp_ratio = kwargs.get("mlp_ratio")

        self.dense1 = nn.Linear(self.channel, self.channel * self.mlp_ratio)
        self.dense2 = nn.Linear(self.channel * self.mlp_ratio, self.channel)
        self.gelu = nn.GELU()
        self.atten = Attention(**kwargs)

    def forward(self, inputs):
        # inputs.shape = (batch, length, channel)
        skip = inputs
        # attention
        results = self.atten(inputs)
        results = skip + results
        # results.shape = (batch, length, channel)
        # mlp with skip connection
        skip = results
        results = self.dense1(results)
        # results.shape = (batch, length, channel * mlp_ratio)
        results = self.dense2(results)
        # results.shape = (batch, length, channel)
        results = skip + results
        return results


class Extractor(nn.Module):
    def __init__(self, **kwargs):
        super(Extractor, self).__init__()
        self.channel = kwargs.get("channel")
        self.depth = kwargs.get("depth")

        self.gelu = nn.GELU()
        self.dense1 = nn.Linear(self.channel, self.channel)
        self.dense2 = nn.Linear(self.channel, self.channel)

        self.layer_blocks = nn.ModuleList([ABlock(**kwargs) for _ in range(self.depth)])
        self.head = nn.Linear(self.channel, self.channel, bias=False)

    def forward(self, inputs):
        # batch = inputs.shape[0]
        # # inputs.shape = (batch, seq, channel)
        results = inputs
        results = self.dense1(inputs)
        # results.shape = (batch, seq, channel)
        results = self.gelu(results)
        # do attention only when the feature shape is small enough
        for i in range(self.depth):
            # result.shape = (batch, seq, channel)
            results = self.layer_blocks[i](results)
            # results.shape = (batch, seq, channel)
        results = self.dense2(results)
        # results.shape = (batch, seq, channel)
        results = self.head(results)
        # results.shape = (batch, seq, channel)
        return results


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.predictor = Extractor(
            channel=4,
            seq=27,
            num_heads=3,
            depth=3,
            mlp_ratio=4,
            **kwargs,
        )

        # input size = torch.Size([1, 2, 3, 3, 3])
        self.cnn = nn.Sequential(
            nn.Conv3d(4, 32, 2),
            nn.GELU(),
            # output size = torch.Size([1, 32, 2, 2, 2])
            nn.Conv3d(32, 108, 2),
            nn.GELU(),
            # output size = torch.Size([1, 128, 1, 1, 1])
        )

        self.fc = nn.Sequential(
            nn.Linear(108, 108),
            nn.GELU(),
            nn.Linear(108, 108),
            nn.GELU(),
            nn.Linear(108, 108),
            nn.GELU(),
            nn.Linear(108, 1),
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # x = x.reshape(-1, 4, 27)
        # x = torch.permute(x, (0, 2, 1))
        # x = self.predictor(x)
        # x = torch.permute(x, (0, 2, 1))

        t = x[:, [0], CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] + ESP
        x = x.reshape(-1, 4 * 27)
        x = x / t
        x = x.reshape(-1, 4, 3, 3, 3)
        x = self.cnn(x)

        x = x.reshape(-1, 108)
        x = self.fc(x)
        x = x * t
        return x
