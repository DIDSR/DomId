import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Net_MNIST(nn.Module):
    def __init__(self, y_dim, img_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flat_size = self.probe(img_size)
        self.fc1 = nn.Linear(self.flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, y_dim)

    def probe(self, img_size):
        x = torch.rand(2, 3, img_size, img_size)
        x = self.conv_op(x)
        list_size = list(x.shape[1:])
        flat_size = np.prod(list_size)
        return flat_size

    def conv_op(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self.conv_op(x)
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_Net_MNIST():
    import torch

    x = torch.rand(2, 3, 28, 28)
    model = Net_MNIST(2, 28)
    model(x)
