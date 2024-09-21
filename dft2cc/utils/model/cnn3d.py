# An 3d cnn model
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self):
        super().__init__()
        # input size = torch.Size([1, 4, 3, 3, 3])
        self.cnn = nn.Sequential(
            nn.Conv3d(4, 64, 3),
            nn.ReLU(),
            # nn.MaxPool3d(1, 1),
        )
        # output size = torch.Size([1, 16, 1, 1, 1])

        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
