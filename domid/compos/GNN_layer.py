import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):
    def __init__(self, in_features, out_features, device):
        import pdb; pdb.set_trace()
        super(GNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros([in_features, out_features], dtype=torch.float, device=device))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, activation=torch.nn.ReLU()):
        # if len(features.shape)>2:
        #     features = torch.flatten(features, 1,-1)
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if activation:
            output = activation(output)

        return output

