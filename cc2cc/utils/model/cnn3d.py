# An 3d cnn model
import torch
import torch.nn as nn
import torch.nn.functional as F

from cc2cc.utils.env_var import CUBE_USE, CUBE_USE_MIDDLE


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self):
        super().__init__()
        # input size = torch.Size([1, 2, 3, 3, 3])
        self.cnn = nn.Sequential(
            nn.Conv3d(4, 32, 2),
            nn.ReLU(),
            # output size = torch.Size([1, 32, 2, 2, 2])
            nn.Conv3d(32, 128, 2),
            nn.ReLU(),
            # output size = torch.Size([1, 128, 1, 1, 1])
        )

        self.dropout = nn.Dropout(p=0.1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # exc = x[:, 0, CUBE_USE_MIDDLE, CUBE_USE_MIDDLE, CUBE_USE_MIDDLE]
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x
