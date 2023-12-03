import torch
import torch.nn as nn

class LinearDynamicalSystem(nn.Module):
    def __init__(self, A, timestep):
        super(LinearDynamicalSystem, self).__init__()

        self.A = A
        self.timestep = timestep

        self.weight = nn.Parameter(torch.randn(A.shape[0], A.shape[1]))
        self.bias = nn.Parameter(torch.randn(A.shape[1]))

        # index mask representing where A is not zero
        idx = torch.where(self.A + torch.eye(self.A.shape[0]) == 0)
        # mask weights by adjacency matrix A, except for the diagonal
        self.weight.data[idx] = 0

    def forward(self, x):
        dxdt = x @ (self.A * self.weight).T
        x = x + dxdt * self.timestep
        # remove the bottom row
        x = x[:-1]
        return x
