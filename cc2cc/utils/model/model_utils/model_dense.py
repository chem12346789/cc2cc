from torch import nn


class DenseNet(nn.Module):
    """
    DenseNet module.
    """

    def __init__(self, **kwargs):
        super(DenseNet, self).__init__()
        self.d_model = kwargs.get("d_model")
        self.mlp = kwargs.get("mlp")
        self.depth = kwargs.get("depth")
        self.drop_rate = kwargs.get("drop_rate")
        self.if_skip_connection_dense = kwargs.get("if_skip_connection_dense")
        self.dense_actv = kwargs.get("dense_actv")
        self.dense_normal = kwargs.get("dense_normal")

        self.sizes = [self.d_model] + [self.mlp] * (self.depth - 1) + [1]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(self.sizes, self.sizes[1:])
            ]
        )

        if self.dense_actv == "relu":
            self.actv_fn = nn.ReLU()
        elif self.dense_actv == "gelu":
            self.actv_fn = nn.GELU()
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
