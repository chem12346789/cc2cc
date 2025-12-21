from torch import nn


class DenseNet(nn.Module):
    """
    DenseNet module.
    """

    def __init__(self, **kwargs):
        super(DenseNet, self).__init__()
        d_model = kwargs.get("d_model")
        mlp = kwargs.get("mlp")
        depth = kwargs.get("depth")
        dense_bias = kwargs.get("dense_bias", True)
        dense_actv = kwargs.get("dense_actv", "gelu")
        dense_normal = kwargs.get("dense_normal", "")
        self.if_skip_connection_dense = kwargs.get("if_skip_connection_dense", True)

        if dense_actv == "relu":
            self.actv_fn = nn.ReLU()
        elif dense_actv == "gelu":
            self.actv_fn = nn.GELU()
        elif dense_actv == "mish":
            self.actv_fn = nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {dense_actv}")

        self.sizes = [d_model] + [mlp] * (depth - 1) + [1]
        if dense_normal == "layer":
            self.norm = nn.ModuleList(
                [nn.LayerNorm(i_size) for i_size in self.sizes[:-2]]
            )
        elif dense_normal == "rms":
            self.norm = nn.ModuleList(
                [nn.RMSNorm(i_size) for i_size in self.sizes[:-2]]
            )
        else:
            self.norm = nn.ModuleList([nn.Identity() for _ in self.sizes[:-2]])
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size, bias=dense_bias)
                for input_size, output_size in zip(self.sizes, self.sizes[1:])
            ]
        )

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
            if self.if_skip_connection_dense:
                if self.sizes[i] == self.sizes[i + 1]:
                    x = x + skip
        return x
