import torch
from e3nn import o3
from e3nn.nn import Gate

from cc2cc.utils.env_var import EDGE_SIZE, EDGE_LEN, CUBE_MIDDLE

# E3NN-based octahedral equivariant network for 3D scalar field prediction.

# This module implements an E3NN network that processes 27 points in a cube,
# each with 1 scalar channel, and outputs 27 scalars at the center point.
# The network is equivariant to full octahedral symmetry.

# Input coordinates: 27 points in a cube from (-0.01, -0.01, -0.01) to (0.01, 0.01, 0.01)
# Output: 27 invariants at the center point.


class E3nn(torch.nn.Module):

    def __init__(self, cube_type, cube_size, input_level, lmax):
        super().__init__()

        self.input_level = input_level
        self.cube_type = cube_type
        self.cube_size = cube_size
        self.lmax = lmax

        if self.cube_type == "cube":
            edge_vec = torch.zeros(
                (EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
                device="cpu",
                dtype=torch.get_default_dtype(),
            )
            for i in range(EDGE_SIZE):
                for j in range(EDGE_SIZE):
                    for k in range(EDGE_SIZE):
                        edge_vec[i, j, k, 0] = (i - CUBE_MIDDLE) * EDGE_LEN
                        edge_vec[i, j, k, 1] = (j - CUBE_MIDDLE) * EDGE_LEN
                        edge_vec[i, j, k, 2] = (k - CUBE_MIDDLE) * EDGE_LEN
        else:
            raise NotImplementedError("Only cube type is implemented.")
        edge_vec = edge_vec.reshape(EDGE_SIZE**3, 3)

        irreps_input = o3.Irreps(f"{self.input_level}x0e")
        hidden_irreps = o3.Irreps(
            "+".join([f"{self.input_level}x{i}e" for i in range(self.lmax + 1)])
        )
        # hidden_irreps = o3.Irreps(
        #     "+".join(
        #         [
        #             f"{self.input_level}x{i}{'e' if i % 2 == 0 else 'o'}"
        #             for i in range(self.lmax + 1)
        #         ]
        #     )
        # )
        irreps_output = o3.Irreps(f"{self.cube_size}x{0}e")

        irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        sh = o3.spherical_harmonics(
            irreps_sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_input,
            irreps_sh,
            hidden_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        self.tensor_square = o3.TensorSquare(
            hidden_irreps,
            irreps_out=irreps_output,
            irrep_normalization="component",
            path_normalization="element",
        )

        # uniform_ initialization for the tensor product weights
        with torch.no_grad():
            self.tp1.weight.uniform_(-1, 1)
            self.tensor_square.weight.uniform_(-1, 1)

        self.register_buffer("edge_vec", edge_vec, persistent=False)
        self.register_buffer("sh", sh, persistent=False)

    def forward(self, f_in):
        # f_in shape: [CUBE_SIZE**3, 4]
        h_local = self.tp1(f_in, self.sh)
        # f_hidden shape: [CUBE_SIZE**3, (lmax+1)**2]
        h_global = h_local.sum(dim=-2, keepdim=True)
        # f_hidden shape: [1, (lmax+1)**2]
        f_out = self.tensor_square(h_global)
        # f_out shape: [1, CUBE_SIZE**3]
        return f_out


@torch.no_grad()
def test_x_reflection(model):
    model.eval()

    x = torch.randn(
        EDGE_SIZE,
        EDGE_SIZE,
        EDGE_SIZE,
        model.input_level,
        device=model.sh.device,
        dtype=model.sh.dtype,
    )

    x_reflected = torch.flip(x, dims=[0])

    y1 = model(x.reshape(EDGE_SIZE**3, -1))
    y2 = model(x_reflected.reshape(EDGE_SIZE**3, -1))

    error = (y1 - y2).abs().max()

    print("Maximum invariant error:", error.item())

    assert torch.allclose(
        y1,
        y2,
        atol=1e-5,
        rtol=1e-5,
    )


@torch.no_grad()
def test_xy_swap(model):
    model.eval()

    x = torch.randn(
        EDGE_SIZE,
        EDGE_SIZE,
        EDGE_SIZE,
        model.input_level,
        device=model.sh.device,
        dtype=model.sh.dtype,
    )

    x_swapped = x.transpose(0, 1)

    y1 = model(x.reshape(EDGE_SIZE**3, -1))
    y2 = model(x_swapped.reshape(EDGE_SIZE**3, -1))

    error = (y1 - y2).abs().max()

    print("Maximum invariant error:", error.item())

    assert torch.allclose(
        y1,
        y2,
        atol=1e-5,
        rtol=1e-5,
    )
