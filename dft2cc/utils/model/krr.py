import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Kernel ridge regression model using the Gaussian kernel
    Use the backward method to optimise the hyperparameters
    output = \\sum kernel_i * weight_i
    kernel_i = exp(-||x - input_vector_i||^2 / sigma)
    """

    def __init__(self, train_vector):
        super().__init__()

        self.input_size = 4
        self.output_size = 1
        # train_vector size = torch.Size([ntrain, 4])
        self.train_vector = train_vector
        # train_vector size = torch.Size([1, ntrain, 4])
        self.train_vector = self.train_vector.unsqueeze(0)

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.weight = nn.Parameter(torch.zeros(len(self.train_vector)))

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # input size = torch.Size([N, 4])
        # Compute the kernel matrix
        if x.dim() == 2:
            for i in range(x.shape[0]):
                # x[i] size = torch.Size([4])
                kernel_i = torch.exp(
                    -torch.sum((x[i] - self.train_vector) ** 2, dim=-1) / self.sigma
                )
                if i == 0:
                    kernel = kernel_i
                else:
                    kernel = torch.cat((kernel, kernel_i), dim=0)
        else:
            kernel = torch.exp(
                -torch.sum((x - self.train_vector) ** 2, dim=-1) / self.sigma
            )

        # kernel size = torch.Size([N, ntrain])

        return torch.sum(kernel * self.weight, dim=-1)
