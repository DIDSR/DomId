import torch
import torch.nn as nn


class DECClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0, device = 'cpu'):
        """
         :param n_clusters: The number of clusters.
         :param hidden: The size of the hidden layer.
        :param cluster_centers: The initial cluster centers.
        :param alpha: The alpha parameter for the Student's t-distribution.
        :param device: The device to use (e.g. 'cpu', 'cuda').
        """
        super(DECClusteringLayer, self).__init__()
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
        """
        Performs forward propagation on the ClusteringLayer.
        Corresponds to equation (1) from the paper.

        :param x:
        :return t_dist:The soft cluster assignments.
        """

        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist