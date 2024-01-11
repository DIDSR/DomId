import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GNNLayer(Module):
    def __init__(self, in_features, out_features, device):

        super(GNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros([in_features, out_features], dtype=torch.float, device=device))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, activation=torch.nn.ReLU()):
        """
        :param features: features from specific layer of the encoder
        :param adj: adjecency matrix from the constructed graph
        :param activation:
        :return: hidden layer of GNN
        """
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if activation:
            output = activation(output)

        return output
