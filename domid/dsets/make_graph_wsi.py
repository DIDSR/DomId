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
from domid.dsets.make_graph import GraphConstructor

class GraphConstructorWSI(GraphConstructor):
    """
    Class to construct graph from features from WSI images.
    This is only used in training for SDCN model and for WSI dataset.
    """
    def __init__(self, graph_method):
        super(GraphConstructorWSI, self).__init__()
        self.graph_method = graph_method



    def distance_calc_wsi(self, features, coordinates=None):
        """
        This function is used to calculate distance between features.
        :param features: the batch of features from the dataset
        :param graph_method: the method to calculate distance between features
        :param coordinates: if the image(patch in the batch) has the coordinates specified, then the distance between can be calculated based on the coordinates
        :return: distance matrix between features of the batch of images with the shape of (num_img, num_img)
        """

        if self.graph_method=='patch_distance':
            num_coords = len(coordinates)
            dist = np.zeros((num_coords, num_coords))

            for i in range(num_coords):
                for j in range(i, num_coords):
                    distance = np.sqrt((coordinates[i][0] - coordinates[j][0]) ** 2+(coordinates[i][1] - coordinates[j][1]) ** 2)
                    dist[i, j] = distance
                    dist[j, i] = distance
        return dist
    
    
    def connection_calc(self, features, region_labels, topk=7):
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

            coordinates = [[int(reg_lab.split('_')[2])+int(reg_lab.split('_')[-1][:-4]),int(reg_lab.split('_')[3])+int(reg_lab.split('_')[-2][2:])] for reg_lab in region_labels]
            d = self.distance_calc_wsi(features, coordinates) #within each region calculate distance between patches
            dist.append(d)
        else:
            dist.append(self.distance_calc_wsi(features))
        
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
    


    def construct_graph(self, features, img_ids,experiment_folder ):
        """
        This function is used to construct the graph for all the batches of dataset. This is called in the trainer function of SDCN model.
        :param dataset: dataset contraining all the batch of data (or no batched data)
        :param graph_method: graph construction method
        :return: the adjacency matrix for one batche of data
        """

        coordinates = ['_'.join(img_id.split('/')[-1].split('_')[-8:]) for img_id in img_ids]
        num_features =  features.shape[0]
        topk = 7 #topk connections for each image

        
        dist, inds, connection_pairs = self.connection_calc(features,coordinates, topk = topk)
        adj_mx = self.mk_adj_mat(num_features, connection_pairs)
        sparse_mx = self.sparse_mx_to_torch_sparse_tensor(adj_mx)
        return adj_mx, sparse_mx
