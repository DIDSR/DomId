import torch
import torch.nn as nn
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0, device = 'cpu'):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_clusters, self.hidden,dtype=torch.float).to(device)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)
    def forward(self, x):

        # x here is z in the equation 1


        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)

        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist