import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from domid.compos.GNN_layer import GNNLayer


class GNN(Module):
    def __init__(self, n_input, n_enc_1, n_enc_2, n_enc_3, n_z, n_clusters, device):
        super(GNN, self).__init__()

        self.gnn_1 = GNNLayer(n_input, n_enc_1, device)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2, device)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3, device)
        self.gnn_4 = GNNLayer(n_enc_3, n_z, device)
        self.gnn_5 = GNNLayer(n_z, n_clusters, device)

    def _flatten_if_needed(self, x):
        return torch.flatten(x, 1, -1) if len(x.shape) > 2 else x

    def forward(self, x, adj, tra1, tra2, tra3, z, sigma=0.5):

        x, tra1, tra2, tra3 = map(self._flatten_if_needed, (x, tra1, tra2, tra3))
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, activation=False)

        return h
