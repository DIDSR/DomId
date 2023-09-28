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
import os

class GraphConstructor():
    """
    Class to construct graph from features. This is only used in training for SDCN model.
    """
    # def parse_name(self, name):
    #     sub_num = name.split('-')[1]
    #     region = name.split('_')[-3]
    #     return sub_num+'_'+region
    #
        
    def get_features_labels(self, dataset):
        """
        This funciton is used to get features and labels from dataset.
        :param dataset: Image dataset that can be batched or unbatched
        :return: X: features from the image (flattened images), labels: domain labels, region_labels: region labels if the dataset is WSI images
        """
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
        """
        Row-normalize sparse matrix which is used to calculate the distance for normalized cosine method.
        :param mx: sparse matrix
        :return: row-normalized sparse matrix
        """
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
    def distance_calc(self, features, graph_method, coordinates=None):
        """
        This function is used to calculate distance between features.
        :param features: the batch of features from the dataset
        :param graph_method: the method to calculate distance between features
        :param coordinates: if the image(patch in the batch) has the coordinates specified, then the distance between can be calculated based on the coordinates
        :return: distance matrix between features of the batch of images with the shape of (num_img, num_img)
        """
        if graph_method == 'heat':
            dist = -0.5 * pair(features) ** 2
            dist = np.exp(dist)
        elif graph_method == 'cos':
            features[features > 0] = 1
            dist = np.dot(features, features.T)
        elif graph_method == 'ncos':
            features[features > 0] = 1
            features = normalize(features, axis=1, norm='l1')
            dist = np.dot(features, features.T)
        elif graph_method=='patch_distance':
            num_coords = len(coordinates)
            dist = np.zeros((num_coords, num_coords))

            for i in range(num_coords):
                for j in range(i, num_coords):
                    distance = np.sqrt((int(coordinates[i][0]) - int(coordinates[j][0])) ** 2+(int(coordinates[i][1]) - int(coordinates[j][1])) ** 2)
                    dist[i, j] = distance
                    dist[j, i] = distance
        return dist
    
    
    def connection_calc(self, features, region_labels,graph_method, topk=7):
        """
        This function is used to calculate the connection pairs between images for all the batches of dataset.
        :param features: flattened image from the batch of dataset
        :param region_labels: if dataset contains spacial information between images, then the region labels can be used to calculate the distance between images
        :param graph_method: graph method to calculate the distance between images
        :param topk: number of connections per image
        :return: indecies of top k connections per each image in the batch (shape: (num_img*topk, 2))
        """
        dist = []
        if len(region_labels)>0:
            # if WSI dataset, then split the features into number of regions per batch
            region_names = [reg_lab.split('_')[0] for reg_lab in region_labels]
            num_regions = len(set(region_names))
            coordinates = [[reg_lab.split('_')[-2][2:],reg_lab.split('_')[-1][:-4]] for reg_lab in region_labels]
            features=np.array_split(features, num_regions)    #FIXME 3 is the number of regions per batch (we are not connecting between regions)
            coordinates = np.array_split(coordinates,num_regions)
            for feat,coord in zip(features, coordinates):
                d = self.distance_calc(feat, graph_method, coord) #within each region calculate distance between patches
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
        """
        This function is used to make the adjacency matrix for the graph for each batch of dataset.
        :param n:
        :param connection_pairs:
        :return:
        """
 
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
    

    def construct_graph(self, dataset, graph_method,experiment_folder ):
        """
        This function is used to construct the graph for all the batches of dataset. This is called in the trainer function of SDCN model.
        :param dataset: dataset contraining all the batches of data (or no batched data)
        :param graph_method: graph construction method
        :return: the adjacency matrix for all the batches of data
        """
        
        adj_matricies = []
        features, labels, region_labels = self.get_features_labels(dataset)
        batch_num = features.shape[0]
        num_features =  features.shape[1]
        topk = 7 #topk connections for each image

        for i in range(0, batch_num):
            dist, inds, connection_pairs = self.connection_calc(features[i, :, :],region_labels[i], graph_method, topk = topk)
#             try:
#                 connect_path = os.path.join('../notebooks/', experiment_folder)+"/connection_pairs_"+str(i)+".pkl"
#                 feat_path = os.path.join('../notebooks/', experiment_folder)+"/features_"+str(i)+".pkl"
#                 label_path = os.path.join('../notebooks/',experiment_folder)+"/labels_"+str(i)+".pkl"
#                 with open(connect_path, "wb") as file:
#                     pickle.dump(connection_pairs, file)

#                 with open(feat_path, "wb") as file:
#                     pickle.dump(features[i, :, :], file)

#                 with open(label_path, "wb") as file:
#                     pickle.dump(labels[i, :], file)
            

            adj_mat = self.mk_adj_mat(num_features, connection_pairs)
            adj_matricies.append(adj_mat)
        return adj_matricies
