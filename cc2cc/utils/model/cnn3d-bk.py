# An 3d cnn model
import torch
import torch.nn as nn

from cc2cc.utils.env_var import CUBE_MIDDLE

ESP = torch.finfo(torch.float32).eps


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, **kwargs):
        super().__init__()

        # input size = torch.Size([1, 2, 3, 3, 3])
        self.cnn = nn.Sequential(
            nn.Conv3d(4, 32, 2),
            nn.GELU(),
            # output size = torch.Size([1, 32, 2, 2, 2])
            nn.Conv3d(32, 128, 2),
            nn.GELU(),
            # output size = torch.Size([1, 128, 1, 1, 1])
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        rho0 = x[:, [0], :, :, :]
        rho1 = x[:, [1], :, :, :]
        rho = rho0 + rho1
        rho_center = rho[:, 0, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE]
        rho_center = rho_center.reshape(-1, 1)

        rho_lda = (rho + ESP) ** (1 / 3)
        rho_spin = (rho0 - rho1) / (rho + ESP)
        xi = (1 + rho_spin + ESP) ** (4 / 3) + (1 - rho_spin + ESP) ** (4 / 3)

        t = torch.cat([rho_lda, xi, x[:, [2], :, :, :], x[:, [3], :, :, :]], dim=1)
        t = self.cnn(t)
        t = t.reshape(t.size(0), 128)
        # x = self.dropout(x)
        t = self.fc(t)
        t = t * rho_center
        return t
