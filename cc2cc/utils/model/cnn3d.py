# An 3d cnn model
import torch

import torch.nn as nn

ESP = torch.finfo(torch.float32).eps
LEN = 4


class Model(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self):
        super().__init__()
        # input size = torch.Size([1, 2, 3, 3, 3])
        self.cnn = nn.Sequential(
            nn.Conv3d(LEN, 32, 2),
            nn.GELU(),
            # output size = torch.Size([1, 32, 2, 2, 2])
            nn.Conv3d(32, 128, 2),
            nn.GELU(),
            # output size = torch.Size([1, 128, 1, 1, 1])
        )

        self.dropout = nn.Dropout(p=0.1)

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

        t = torch.empty(
            (
                x.shape[0],
                LEN,
                x.shape[2],
                x.shape[3],
                x.shape[4],
            ),
            dtype=x.dtype,
            device=x.device,
        )

        rho0 = x[:, 0, :, :, :]
        rho1 = x[:, 1, :, :, :]

        rho_lda = (rho0 + rho1) ** (1 / 3)
        t[:, 0, :, :, :] = rho_lda

        rho_spin = (rho0 - rho1) / (rho0 + rho1 + ESP)
        t[:, 1, :, :, :] = (1 + rho_spin) ** (4 / 3) + (1 - rho_spin) ** (4 / 3)
        t[:, 2, :, :, :] = (
            x[:, 2, :, :, :] + 2 * x[:, 3, :, :, :] + x[:, 4, :, :, :]
        ) / (rho_lda**4 + ESP)

        if LEN == 4:
            t[:, 3, :, :, :] = (x[:, 5, :, :, :] - x[:, 6, :, :, :]) / (
                rho_lda**5 + ESP
            )

        t = torch.log(t + ESP)
        t = self.cnn(t)
        t = t.reshape(x.size(0), -1)
        # x = self.dropout(x)
        t = self.fc(t)
        return t
