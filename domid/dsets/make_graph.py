import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import pickle

class GraphConstructor():
    def parse_name(self, name):
        sub_num = name.split('-')[1]
        region = name.split('_')[-3]
        return sub_num+'_'+region
        
        
    def get_features_labels(self, dataset):
        num_batches = len(dataset)
        num_img,i_c, i_w, i_h =next(iter(dataset))[0].shape
        X = torch.zeros((num_batches, num_img, i_c * i_w * i_h))
        labels = torch.zeros((num_batches, num_img, 1))
        region_labels = [] #torch.zeros((num_batches, num_img,1))
        counter = 0
        for tensor_x, vec_y, vec_d, inj_tensor, img_ids in dataset:

            X[counter, :, :]=torch.reshape(tensor_x, (tensor_x.shape[0], tensor_x.shape[1]*tensor_x.shape[2]*tensor_x.shape[3]))
            labels[counter, :, 0]=torch.argmax(vec_d, dim=1)
            if 'aperio' in img_ids[0]:
                regions = ['_'.join(img_id.split('/')[-1].split('_')[-8:]) for img_id in img_ids]
                region_labels.append(regions) #, dtype=torch.string)
            else:
                print('not wsi images')
                regions=[]
                region_labels.append(regions)
                
            counter+=1

        return X.type(torch.float32), labels.type(torch.int32), region_labels
    def normalize(self, mx): #FIXME move to utils
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
    def distance_calc(self, features,coordinates=None, method = 'heat'):
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
        elif method=='patch_distance':
            num_coords = len(coordinates)
            dist = np.zeros((num_coords, num_coords))

            for i in range(num_coords):
                for j in range(i, num_coords):
                    distance = np.sqrt((int(coordinates[i][0]) - int(coordinates[j][0])) ** 2+(int(coordinates[i][1]) - int(coordinates[j][1])) ** 2)
                    dist[i, j] = distance
                    dist[j, i] = distance
        return dist
    
    
    def connection_calc(self, features, labels, region_labels, topk=7,method='heat'):
        dist = []
        if len(region_labels)>0:
            method='patch_distance'
            # region_labels = np.array(region_labels)
            # sorted_indices = np.argsort(region_labels)
            # features = features[sorted_indices, :]
            # labels = labels[sorted_indices, :]
            # region_labels = region_labels[sorted_indices]
            coordinates = [[reg_lab.split('_')[-2][2:],reg_lab.split('_')[-1][:-4]] for reg_lab in region_labels]
            features=np.array_split(features, 3)    
            coordinates = np.array_split(coordinates,3)
            for feat,coord in zip(features, coordinates):
                d = self.distance_calc(feat, coord, method)
                dist.append(d)
        else:
            dist.append(self.distance_calc(features))
        
        connection_pairs = [] 
        inds = []
        counter =0
        for region in dist:
            for i in range(region.shape[0]):
                ind = np.argpartition(region[i, :], -(topk+1))[-(topk+1):]
                ind = ind+np.ones(len(ind))*counter*region.shape[0]
                ind = ind.astype(np.int32)
                inds.append(ind) #each patch's 10 connections
            counter+=1
        for i, v in enumerate(inds):
            for vv in v:
                if vv == i:
                    pass
                else:
                    connection_pairs.append([i, vv])
                    
                    
 
        return dist, inds, connection_pairs
    def mk_adj_mat(self, n, connection_pairs):
 
        idx = np.array([i for i in range(n)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.array(connection_pairs,  dtype=np.int32) #features #np.genfromtxt(path, dtype=np.int32)
        edges_mapped = [idx_map.get(val, -1) for val in edges_unordered.flatten()]
        if -1 in edges_mapped:
            print("Error: Some keys in edges_unordered do not exist in idx_map.")
        else:
            edges = np.array(edges_mapped, dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        adj = adj + sp.eye(adj.shape[0])
        adj = self.normalize(adj)
        return adj
    
    def plot_graph(self, edges, labels, bs):
        
        num_nodes = len(edges)
        
        # for i in range(num_nodes):
        #     for j in range(i + 1, num_nodes):  # Exclude diagonal and symmetric entries
        #         if adjacency_matrix[i, j] != 0:
        #             edges.append((i, j))
        ed= [(edge[0], edge[1]) for edge in edges]
        #labels = {0: '0', 1: '1', 2: '2', 3: '3', 4:'4', 5:'5', 6:'6'}
        labels_color= {0:'blue', 1:'navy', 2:'green',3:'yellow', 4:'orange', 5: 'peach' }
        graph = nx.Graph(ed)
        node_colors = [labels_color[labels[node]] if labels[node]>1 else 'purple' for node in graph.nodes()]
        pos = nx.spring_layout(graph)  # Specify the layout for node positions
        nx.draw_networkx(graph, pos=pos, node_color=node_colors, with_labels=True)
        plt.show()
        plt.savefig("../../graph_bs_"+str(bs)+".png")
        plt.close()


    
    def construct_graph(self, dataset):
        
        adj_matricies = []
        features, labels, region_labels = self.get_features_labels(dataset)
        batch_num = features.shape[0]
        num_features =  features.shape[1]
        #distance_batches = np.zeros((batch_num, num_features, num_features))
        topk = 7

        for i in range(0, batch_num):
            dist, inds, connection_pairs = self.connection_calc(features[i, :, :], labels[i, :],region_labels[i], topk = topk)
            connect_path = "../../graph_vis_data/connection_pairs_"+str(i)+".pkl" 
            feat_path = "../../graph_vis_data/features_"+str(i)+".pkl"
            label_path = "../../graph_vis_data/labels_"+str(i)+".pkl"
            with open(connect_path, "wb") as file:
                pickle.dump(connection_pairs, file)
                
            with open(feat_path, "wb") as file:
                pickle.dump(features[i, :, :], file)
                
            with open(label_path, "wb") as file:
                pickle.dump(labels[i, :], file)
            

            adj_mat = self.mk_adj_mat(num_features, connection_pairs)
            adj_matricies.append(adj_mat)
#             pdb.set_trace()
#             self.plot_graph(connection_pairs, labels[i, :], i)
            
            

       
        #adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        # # f.close()
        # print(counter)
        # print('error rate: {}'.format(counter / (num_features * topk)))
        return adj_matricies
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