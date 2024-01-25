import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import pairwise_distances as pair
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class GraphConstructor:

    """
    Class to construct graph from features. This is only used in training for SDCN model.
    """

    def __init__(self, graph_method, topk=7):
        """
        Initializer of GraphConstructor.
        :param graph_method: the method to calculate distance between features; one of 'heat', 'cos', 'ncos'.
        :param topk: number of connections per image
        """
        self.graph_method = graph_method
        self.topk = topk

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):  # FIXME move to utils
        """Convert a scipy sparse matrix to a torch sparse
        tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_features_labels(self, dataset):
        """
        This funciton is used to get features and labels from dataset.
        :param dataset: Image dataset that can be batched or unbatched
        :return: X: features from the image (flattened images), labels: domain labels, region_labels: region labels if the dataset is WSI images
        """
        num_batches = len(dataset)
        num_img, i_c, i_w, i_h = next(iter(dataset))[0].shape
        X = torch.zeros((num_batches, num_img, i_c * i_w * i_h))
        labels = torch.zeros((num_batches, num_img, 1))
        counter = 0
        for tensor_x, vec_y, vec_d, inj_tensor, img_ids in dataset:
            X[counter, :, :] = torch.reshape(tensor_x, (tensor_x.shape[0], i_c * i_w * i_h))
            labels[counter, :, 0] = torch.argmax(vec_d, dim=1)
            counter += 1

        return X.type(torch.float32), labels.type(torch.int32)

    def normalize(self, mx):  # FIXME move to utils
        """
        Row-normalize sparse matrix which is used to calculate the distance for normalized cosine method.
        :param mx: sparse matrix
        :return: row-normalized sparse matrix
        """

        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0  # i.e., when row sum is 0, we will keep that row at 0 in themultiplication below
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def distance_calc(self, features):
        """
        This function is used to calculate distance between features.
        :param features: the batch of features from the dataset
        :return: distance matrix between features of the batch of images with the shape of (num_img, num_img)
        """
        if self.graph_method == "heat":
            dist = -0.5 * pair(features) ** 2
            dist = np.exp(dist)
        elif self.graph_method == "cos":
            features[features > 0] = 1
            dist = np.dot(features, features.T)
        elif self.graph_method == "ncos":
            features[features > 0] = 1
            features = normalize(features, axis=1, norm="l1")
            dist = np.dot(features, features.T)

        return dist

    def connection_calc(self, features):
        """
        This function is used to calculate the connection pairs between images for all the batches of dataset.
        :param features: flattened image from the batch of dataset
        :return: indecies of top k connections per each image in the batch (shape: (num_img*self.topk, 2))
        """

        dist = self.distance_calc(features)
        print(dist)
        connection_pairs = []
        inds = []
        for i in range(dist.shape[0]):
            ind = np.argpartition(dist[i, :], -(self.topk + 1))[-(self.topk + 1) :]
            inds.append(ind)

        for i, v in enumerate(inds):
            for vv in v:
                if vv == i:
                    pass
                else:
                    connection_pairs.append([i, vv])
        return dist, inds, connection_pairs

    def mk_adj_mat(self, n, connection_pairs):
        """
        This function is used to make the adjacency matrix for the graph for each batch of dataset.
        :param n: batchsize
        :param connection_pairs: top k connections per each image in the batch (shape: (num_img*self.topk, 2))
        :return:
        """

        idx = np.array([i for i in range(n)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.array(connection_pairs, dtype=np.int32)  # features #np.genfromtxt(path, dtype=np.int32)
        edges_mapped = [idx_map.get(val, -1) for val in edges_unordered.flatten()]
        if -1 in edges_mapped:
            print("Error: Some keys in edges_unordered do not exist in idx_map.")
        else:
            edges = np.array(edges_mapped, dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = self.normalize(adj)

        return adj

    def construct_graph(self, dataset, experiment_folder):
        """
        This function is used to construct the graph for all the batches of dataset. This is called in the trainer function of SDCN model.
        :param dataset: dataset contraining all the batches of data (or no batched data)
        :param graph_method: graph construction method
        :return: the adjacency matrix for all the batches of data
        """

        sparse_matrices = []
        adjacency_matrices = []
        features, domain_labels = self.get_features_labels(dataset)
        batch_num = features.shape[0]
        num_features = features.shape[1]

        for i in range(0, batch_num):
            dist, inds, connection_pairs = self.connection_calc(features[i, :, :])
            adj_mat = self.mk_adj_mat(num_features, connection_pairs)
            adjacency_matrices.append(adj_mat)
            sparse_mx = self.sparse_mx_to_torch_sparse_tensor(adj_mat)
            sparse_matrices.append(sparse_mx)
            if experiment_folder is not None:
                connect_path = (
                    os.path.join("notebooks/", experiment_folder) + "/connection_pairs_" + str(i) + ".pkl"
                )  # FIXME move to zout
                feat_path = os.path.join("notebooks/", experiment_folder) + "/features_" + str(i) + ".pkl"
                label_path = os.path.join("notebooks/", experiment_folder) + "/labels_" + str(i) + ".pkl"
                with open(connect_path, "wb") as file:
                    pickle.dump(connection_pairs, file)

                with open(feat_path, "wb") as file:
                    pickle.dump(features[i, :, :], file)

                with open(label_path, "wb") as file:
                    pickle.dump(domain_labels[i, :], file)
        return adjacency_matrices, sparse_matrices
