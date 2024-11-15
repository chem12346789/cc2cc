# An 3d cnn model
import torch
import torch.nn as nn
import torch.nn.functional as F

from cc2cc.utils.env_var import CUBE_USE


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    if CUBE_USE == 3:

        def __init__(self):
            super().__init__()
            # input size = torch.Size([1, 2, 3, 3, 3])
            self.cnn = nn.Sequential(
                nn.Conv3d(2, 32, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 32, 2, 2, 2])
                nn.Conv3d(32, 256, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 128, 1, 1, 1])
            )

            self.fc = nn.Sequential(
                nn.Linear(256, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    elif CUBE_USE == 5:

        def __init__(self):
            super().__init__()
            # input size = torch.Size([1, 2, 5, 5, 5])
            self.cnn = nn.Sequential(
                nn.Conv3d(2, 8, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 8, 4, 4, 4])
                nn.Conv3d(8, 32, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 32, 3, 3, 3])
                nn.Conv3d(32, 128, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 128, 2, 2, 2])
                nn.Conv3d(128, 256, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 256, 1, 1, 1])
            )

            self.fc = nn.Sequential(
                nn.Linear(256, 16),
                # nn.Linear(512, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
