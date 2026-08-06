import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

# E3NN-based octahedral equivariant network for 3D scalar field prediction.

# This module implements an E3NN network that processes 27 points in a cube,
# each with 1 scalar channel, and outputs 27 scalars at the center point.
# The network is equivariant to full octahedral symmetry.

# Input coordinates: 27 points in a cube from (-0.01, -0.01, -0.01) to (0.01, 0.01, 0.01)
# Output coordinates: 1 point at (0, 0, 0)


class E3nnRadis(torch.nn.Module):
    def __init__(
        self,
        cube_type="cube",
        cube_size=27,
        input_level=4,
        lmax=2,
        n_basis=8,
        out_l=0,
    ):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float64

        if cube_type == "cube":
            edge_vec = torch.zeros(
                (EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
                device=device,
                dtype=dtype,
            )
            self.register_buffer("edge_vec", edge_vec, persistent=False)
            for i in range(EDGE_SIZE):
                for j in range(EDGE_SIZE):
                    for k in range(EDGE_SIZE):
                        self.edge_vec[i, j, k, 0] = (i - CUBE_MIDDLE) * EDGE_LEN
                        self.edge_vec[i, j, k, 1] = (j - CUBE_MIDDLE) * EDGE_LEN
                        self.edge_vec[i, j, k, 2] = (k - CUBE_MIDDLE) * EDGE_LEN
        else:
            raise NotImplementedError("Only cube type is implemented.")
        self.edge_vec = self.edge_vec.reshape(EDGE_SIZE**3, 3)

        irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.sh = o3.spherical_harmonics(
            irreps_sh,
            self.edge_vec,
            normalize=True,
            normalization="norm",
        )

        edge_length = self.edge_vec.norm(dim=-1)
        max_radius = EDGE_SIZE * EDGE_LEN
        self.edge_weight = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=max_radius,
            number=n_basis,
            cutoff=False,
            basis="gaussian",
        )

        irreps_input = o3.Irreps(f"{input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join([f"{max(16 - 4*i, 4)}x{i}e" for i in range(lmax + 1)])
        )
        irreps_output = o3.Irreps(f"{cube_size}x{out_l}e")

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_input,
            irreps_sh,
            hidden_irreps,
            shared_weights=False,
        )
        self.radial_net = torch.nn.Sequential(
            torch.nn.Linear(n_basis, 16, dtype=dtype),
            torch.nn.GELU(),
            torch.nn.Linear(16, self.tp.weight_numel, dtype=dtype),
        )

        self.tp_invariant = o3.FullyConnectedTensorProduct(
            hidden_irreps,
            hidden_irreps,
            irreps_output,
            shared_weights=True,
            internal_weights=True,
        )

        scalar_dim = hidden_irreps[0].dim
        self.scalar_slices = []
        offset = 0
        for mul, ir in hidden_irreps:
            if ir.l == 0:
                self.scalar_slices.append((offset, offset + mul * ir.dim))
            offset += mul * ir.dim
        self.scalar_proj = torch.nn.Linear(
            scalar_dim, cube_size, bias=False, dtype=dtype
        )

        self._init_weights()

    def _init_weights(self):
        # xavier_uniform_ initialization for the tensor product weights
        with torch.no_grad():
            for weight in self.tp_invariant.weight_views():
                mul_1, mul_2, mul_out = weight.shape
                # formula from torch.nn.init.xavier_uniform_
                a = (6 / (mul_1 * mul_2 + mul_out)) ** 0.5
                weight.uniform_(-a, a)

    def forward(self, f_in):
        # f_in shape: [CUBE_SIZE**3, 4]
        weights = self.radial_net(self.edge_weight)  # (27, weight_numel)
        f_hidden = self.tp(f_in, self.sh, weights)
        f_pooled = f_hidden.sum(dim=-2)  # (..., dim_hidden)
        # f_pooled shape: [CUBE_SIZE**3, dim_hidden]
        f_inv = self.tp_invariant(f_pooled, f_pooled)  # (..., cube_size)
        return f_inv.reshape(1, -1)

        f_scalar = torch.cat(
            [f_pooled[..., s:e] for s, e in self.scalar_slices], dim=-1
        )
        f_scalar_out = self.scalar_proj(f_scalar)

        f_out = (f_inv + f_scalar_out).reshape(1, -1)
        # shape: [1, CUBE_SIZE**3]
        return f_out
