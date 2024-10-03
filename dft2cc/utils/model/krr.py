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
        self.n_train = len(self.train_vector)
        self.train_vector = self.train_vector.unsqueeze(0)

        self.weight = nn.Parameter(torch.zeros(self.n_train))
        self.sigma = nn.Parameter(torch.tensor(100.0))
        print(self.weight.shape)
        print(self.train_vector.shape)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        # input size = torch.Size([N, 4])
        # Compute the kernel matrix
        kernel_distance = torch.zeros(x.shape[0], self.n_train, x.shape[1]).to(x.device)
        kernel_distance += x.unsqueeze(1)
        kernel_distance -= self.train_vector
        kernel_distance = torch.sum(kernel_distance**2, dim=-1) / self.sigma
        kernel = torch.exp(-kernel_distance)
        kernel = torch.matmul(kernel, self.weight)
        return kernel
