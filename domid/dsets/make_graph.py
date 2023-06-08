import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
import scipy.sparse as sp
class GraphConstructor():
    def get_features_labels(self, dataset):
        num_img =  len(dataset.dataset)
        X = torch.zeros((num_img, 3 * 16 * 16))
        labels = torch.zeros((num_img, 1))

        counter = 0

        for tensor_x, vec_y, vec_d, *other_vars in dataset:
            X[counter:(counter+vec_y.shape[0]), :]=tensor_x.view(tensor_x.size(0), -1)

            labels[counter:(counter+vec_y.shape[0]), 0]=torch.argmax(vec_y, dim=1)
            counter+=vec_y.shape[0]

        return X.type(torch.float32), labels.type(torch.int32)
    def normalize(self, mx): #FIXME move to utils
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    def construct_graph(self, dataset, method='heat'):

        topk = 10

        features, label = self.get_features_labels(dataset)
        #fname = '../graph/usps_custom_graph.txt'

        num =  features.shape[0]
        dist = None

        if method == 'heat':
            dist = -0.5 * pair(features) ** 2
            dist = np.exp(dist)
        elif method == 'cos':
            features[features > 0] = 1
            dist = np.dot(features, features.T)
        elif method == 'ncos':
            features[features > 0] = 1
            features = normalize(features, axis=1, norm='l1')
            dist = np.dot(features, features.T)

        inds = []
        for i in range(dist.shape[0]):
            ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
            inds.append(ind)

        #f = open(fname, 'w')
        counter = 0
        A = np.zeros_like(dist)
        graph_txt =[] # np.zeros((num, 2))
        for i, v in enumerate(inds):
            mutual_knn = False
            for vv in v:
                if vv == i:
                    pass
                else:
                    if label[vv] != label[i]:
                        counter = counter+1
                    A[i, vv] = vv
                    graph_txt.append([i, vv])


        n, _ = features.shape

        idx = np.array([i for i in range(n)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.array(graph_txt,  dtype=np.int32) #features #np.genfromtxt(path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])
        adj = self.normalize(adj)
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        # f.close()
        print(counter)
        print('error rate: {}'.format(counter / (num * topk)))
        return adj
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx): #FIXME move to utils
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    def load_graph(self, dataset='usps_custom'): #FIXME create a grpah for the dataset? and move to utils

        path = '../graph/{}_graph.txt'.format(dataset)

        data = np.loadtxt('../data/{}.txt'.format(dataset))


        return adj

'''
f = h5py.File('data/usps.h5', 'r')
train = f.get('train')
test = f.get('test')
X_tr = train.get('data')[:]
y_tr = train.get('target')[:]
X_te = test.get('data')[:]
y_te = test.get('target')[:]
f.close()
usps = np.concatenate((X_tr, X_te)).astype(np.float32)
label = np.concatenate((y_tr, y_te)).astype(np.int32)
'''

'''
hhar = np.loadtxt('data/hhar.txt', dtype=float)
label = np.loadtxt('data/hhar_label.txt', dtype=int)
'''
#
# reut = np.loadtxt('data/reut.txt', dtype=float)
# label = np.loadtxt('data/reut_label.txt', dtype=int)

# from domid.dsets.dset_usps import DsetUSPS
# from domid.tasks.task_usps import NodeTaskUSPS
# from domid.arg_parser import parse_cmd_args
# breakpoint()
#
# from domainlab.arg_parser import mk_parser_main
# parser = mk_parser_main()
# args = parser.parse_args(["--tr_d", "1", "2","3", "4", "5", "6", "7", "8", "9", "--dpath", "zout", "--split", "0.8"])
#
# node = NodeTaskUSPS()
# trans = [transforms.Resize((16, 16)), transforms.ToTensor()]
# digit = int(args.tr_d[0])
# dset = DsetUSPS(digit= digit, args = args, list_transforms=trans)
# digit_inds = dset.get_original_indicies()
#
# #dataset = datasets.USPS(root=dpath, train=True, download=True, transform=None)
# features = dataset.data
# features = torch.tensor(features, dtype=torch.float32)
# rgb_tensor= torch.stack([features] * 3, dim=1)
# flattened_features = rgb_tensor.reshape(rgb_tensor.shape[0],rgb_tensor.shape[1]*rgb_tensor.shape[2]*rgb_tensor.shape[3])
#
# print(flattened_features.shape)
# labels = dataset.targets
# construct_graph(flattened_features, labels, 'ncos')
#
# features_file_path = "../data/usps_custom.txt"
#
# # Write the flattened features to the file
# with open(features_file_path, 'w') as file:
#     for feature in flattened_features:
#         file.write(' '.join([str(value.item()) for value in feature]) + '\n')
#
# labels_path = "../data/usps_custom_labels.txt"
#
# # Write the flattened features to the file
# with open(labels_path, 'w') as labels_file:
#     for label in labels:
#         labels_file.write(str(label) + '\n')