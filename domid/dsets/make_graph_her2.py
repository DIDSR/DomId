import numpy as np
import pandas as pd
import torch
from domid.dsets.make_graph import GraphConstructor
import os
import pickle
from sklearn.metrics import pairwise_distances as pair

class GraphConstructorHER2(GraphConstructor):
    """
    Class to construct graph from features from WSI images.
    This is only used in training for SDCN model and for WSI dataset.
    """

    def __init__(self, graph_method, topk=8):
        """
        Initializer of GraphConstructor.
        :param graph_method: the method to calculate distance between features; one of 'heat', 'cos', 'ncos', 'patch_distance'.
        :param topk: number of connections per image
        """
        super().__init__(graph_method, topk)
        self.meta_data_coord = pd.read_csv('../../parsed_HER2.csv')
        
    def get_features_labels(self, dataset):
        """
        This funciton is used to get features and labels from dataset.
        :param dataset: Image dataset that can be batched or unbatched
        :return: X: features from the image (flattened images), labels: domain labels, region_labels: region labels if the dataset is WSI images
        """
        #import pdb;pdb.set_trace()
        num_batches = len(dataset)
        num_img, i_c, i_w, i_h = next(iter(dataset))[0].shape
        X = torch.zeros((num_batches, num_img, i_c * i_w * i_h))
        labels =[]
        counter = 0
        for tensor_x, vec_y, vec_d, inj_tensor, img_ids in dataset:
            X[counter, :, :] = torch.reshape(tensor_x, (tensor_x.shape[0], i_c * i_w * i_h))
            labels.append(img_ids)
            counter += 1

        return X.type(torch.float32), labels

    def distance_calc_wsi(self, features=None, coordinates=None):
        """
        This function is used to calculate distance between features.
        :param features: the batch of features from the dataset
        :param coordinates: if the image(patch in the batch) has the coordinates specified, then the distance between can be calculated based on the coordinates
        :return: distance matrix between features of the batch of images with the shape of (num_img, num_img)
        """
        if self.graph_method == "her_dist":
            num_coords = len(coordinates)
            dist = np.zeros((num_coords, num_coords))
            subject_id = [cc[-1] for cc in coordinates]
            # import pdb;pdb.set_trace()
            dist = -0.5 * pair(features) ** 2
            dist = np.exp(dist)
            
#             features[features > 0] = 1
#             dist = np.dot(features, features.T)
    
#             features[features > 0] = 1
#             features = normalize(features, axis=1, norm="l1")
#             dist = np.dot(features, features.T)
                
       
            # for i in range(num_coords):
            #     for j in range(i, num_coords):
            #         distance = np.sqrt(
            #             (coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2
            #         )
            #         dist[i, j] = distance
            #         dist[j, i] = distance
       
            for jj, target in enumerate(subject_id):
                indices = [i for i, x in enumerate(subject_id) if x == target]
                dist[indices, jj] = -1
                dist[jj, indices] = -1
                
#                 for indx in indices:
#                     distance_calc= np.sqrt(
#                             (coordinates[jj][0] - coordinates[indx][0]) ** 2 + (coordinates[jj][1] - coordinates[indx][1]) ** 2
#                         )
#                     dist[indx, jj]=-distance_calc
#                     dist[jj, indx]=-distance_calc
                
                
#                 for index in indices:
#                     dist[index, jj]=-1
        
        else:
            dist = super().distance_calc(features)
        # import pdb;pdb.set_trace()
        print('DISTANCE \n', dist)
        return dist

    def connection_calc(self, features, region_labels):
        """
        This function is used to calculate the connection pairs between images for all the batches of dataset.
        :param features: flattened image from the batch of dataset
        :param region_labels:  spacial information between patches used to calculate the distance between them (e.g. of the string '1Carcinoma_coord_39100_39573_patchnumber_98_xy_0_0.png')
        :return: indecies of top k connections per each image in the batch (shape: (num_img*self.topk, 2))
        """

        dist = []
        if len(region_labels) > 0:
            # sample region_label that is passed to this function is '1Carcinoma_coord_39100_39573_patchnumber_98_xy_0_0.png'
            # the coordinated of the region would then be (39100, 39573) and the coordinates of the patch would be (0, 0)
            d = self.distance_calc_wsi(features, region_labels)  # within each region calculate distance between patches
            dist.append(d)
        else:
            dist.append(self.distance_calc_wsi(features))

        connection_pairs = []
        inds = []
        counter = 0
        for region in dist:
            for i in range(region.shape[0]):
                ind = np.argpartition(region[i, :], -(self.topk + 1))[-(self.topk + 1) :]
                ind = ind + np.ones(len(ind)) * counter * region.shape[0]
                ind = ind.astype(np.int32)
                inds.append(ind)  # each patch's 10 connections
            counter += 1
        for i, v in enumerate(inds):
            for vv in v:
                if vv == i:
                    pass
                else:
                    connection_pairs.append([i, vv])
        return dist, inds, connection_pairs

    def construct_graph(self, dataset, experiment_folder):
        """
        This function is used to construct the graph for all the batches of dataset. This is called in the trainer function of SDCN model.
        :param features: flattened image from the batch of dataset
        :img_ids:
        :experiment_folder:
        :return: the adjacency matrix for one batch of data
        """

#         print(features[0])
#         print(img_ids[:5])
#         img_id_short = [ii.split("/")[-1] for ii in img_ids]
#         filtered_df = self.meta_data_coord[self.meta_data_coord['img_id'].isin(img_id_short)]

        
#         xx_values = filtered_df['X']
#         yy_values = filtered_df['Y']
        
#         coordinates = list(zip(xx_values, yy_values))
# #         coordinates = ["_".join(img_id.split("/")[-1].split("_")[-8:]) for img_id in img_ids]
#         num_features = features.shape[0]

#         dist, inds, connection_pairs = self.connection_calc(features, coordinates)
#         adj_mx = self.mk_adj_mat(num_features, connection_pairs)
#         sparse_mx = self.sparse_mx_to_torch_sparse_tensor(adj_mx)
        
        
        ################3333
        sparse_matrices = []
        adjacency_matrices = []
        features, img_ids = self.get_features_labels(dataset) #labels == img_ids
        batch_num = features.shape[0]
        num_features = features.shape[1]

        for i in range(0, batch_num):
            # import pdb; pdb.set_trace()
            img_id_short = [ii.split("/")[-1] for ii in img_ids[i]]
            filtered_df = self.meta_data_coord[self.meta_data_coord['img_id'].isin(img_id_short)]
            xx_values = filtered_df['X']
            yy_values = filtered_df['Y']
            sub_num = filtered_df['subject']
            coordinates = list(zip(xx_values, yy_values, sub_num))
            
            dist, inds, connection_pairs = self.connection_calc(features[i, :, :], coordinates)
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
                    pickle.dump(img_ids[i], file)
        # import pdb; pdb.set_trace()
        return adjacency_matrices, sparse_matrices
