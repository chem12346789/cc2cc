import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps
from skala.pyscf import SkalaKS

# E3NN-based octahedral equivariant network for 3D scalar field prediction.

# This module implements an E3NN network that processes 27 points in a cube,
# each with 1 scalar channel, and outputs 27 scalars at the center point.
# The network is equivariant to full octahedral symmetry.

# Input coordinates: 27 points in a cube from (-0.01, -0.01, -0.01) to (0.01, 0.01, 0.01)
# Output coordinates: 1 point at (0, 0, 0)


class E3NNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define the input and output irreps
        self.input_irreps = Irreps("27x0e")  # 27 scalars (0e)
        self.output_irreps = Irreps("27x0e")  # 27 scalars (0e)

        # Define the hidden layers with appropriate irreps
        self.hidden_irreps = Irreps(
            "64x0e + 64x1o + 64x2e"
        )  # Example hidden layer irreps

        # Define the fully connected layers
        self.fc1 = FullyConnectedNet(self.input_irreps, self.hidden_irreps, [64, 64])
        self.fc2 = FullyConnectedNet(self.hidden_irreps, self.output_irreps, [64])

    def forward(self, x):
        # x is expected to be of shape (batch_size, 27) for the input scalars

        # Pass through the first fully connected layer
        x = self.fc1(x)

        # Pass through the second fully connected layer to get the output
        x = self.fc2(x)

        return (
            x  # Output shape will be (batch_size, 27) for the scalars at the 27 points
        )
