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
            # input size = torch.Size([1, 4, 3, 3, 3])
            self.cnn = nn.Sequential(
                nn.Conv3d(4, 64, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 64, 2, 2, 2])
                nn.Conv3d(64, 128, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 128, 1, 1, 1])
            )

            self.fc = nn.Sequential(
                nn.Linear(128, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    elif CUBE_USE == 5:

        def __init__(self):
            super().__init__()
            # input size = torch.Size([1, 4, 5, 5, 5])
            self.cnn = nn.Sequential(
                nn.Conv3d(4, 64, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 64, 4, 4, 4])
                nn.Conv3d(64, 128, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 128, 3, 3, 3])
                nn.Conv3d(128, 256, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 256, 2, 2, 2])
                nn.Conv3d(256, 512, 2),
                nn.ReLU(),
                # output size = torch.Size([1, 512, 1, 1, 1])
            )

            self.fc = nn.Sequential(
                nn.Linear(512, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    elif CUBE_USE == 7:

        def __init__(self):
            super().__init__()
            # input size = torch.Size([1, 4, 7, 7, 7])
            self.cnn = nn.Sequential(
                nn.Conv3d(4, 32, 3),
                nn.ReLU(),
                # output size = torch.Size([1, 32, 5, 5, 5])
                nn.Conv3d(32, 64, 3),
                nn.ReLU(),
                # output size = torch.Size([1, 64, 3, 3, 3])
                nn.Conv3d(64, 128, 3),
                nn.ReLU(),
                # output size = torch.Size([1, 128, 1, 1, 1])
            )

            self.fc = nn.Sequential(
                nn.Linear(128, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    elif CUBE_USE == 9:

        def __init__(self):
            super().__init__()
            # input size = torch.Size([1, 4, 9, 9, 9])
            self.cnn = nn.Sequential(
                nn.Conv3d(4, 32, 3),
                nn.ReLU(),
                # output size = torch.Size([1, 32, 7, 7, 7])
                nn.Conv3d(32, 64, 3),
                nn.ReLU(),
                # output size = torch.Size([1, 64, 5, 5, 5])
                nn.Conv3d(64, 128, 3),
                nn.ReLU(),
                # output size = torch.Size([1, 128, 3, 3, 3])
                nn.Conv3d(128, 256, 3),
                nn.ReLU(),
                # output size = torch.Size([1, 256, 1, 1, 1])
            )

            self.fc = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
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
