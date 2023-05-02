import numpy as np
import torch
from domainlab.utils.perf import PerfClassif
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from domid.compos.predict_basic import Prediction
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import scipy
class PerfSimilarity(PerfClassif):
    """Clustering Performance"""

    def __init__(self):
        super().__init__()

    def encode_array(self, arr):
        """
        This function takes an array as input and encodes any string elements as integers.
        """
        encoded_arr = []
        string_map = {}
        for element in arr:
            if isinstance(element, str):
                if element not in string_map:
                    string_map[element] = len(string_map)
                encoded_arr.append(string_map[element])
            else:
                encoded_arr.append(element)
        return encoded_arr

    def hungarian_algorithm(self, cluster_pred_scalar, cluster_true_scalar):
        """
        This function takes two arrays as input, encodes any string elements to integers,
        and applies the Hungarian Algorithm to find the optimal assignment between the two arrays.
        """

        if len(np.unique(cluster_pred_scalar)) == len(np.unique(cluster_true_scalar)):
            cluster_pred_scalar = [item - 1 for item in cluster_pred_scalar]
            cost = np.zeros((len(np.unique(cluster_pred_scalar)), len(np.unique(cluster_pred_scalar))))
            print(np.unique(cluster_pred_scalar))
            print(np.unique(cluster_true_scalar))

            cost = cost - confusion_matrix(cluster_pred_scalar, cluster_true_scalar)

            # What is the best permutation?
            row_ind, col_ind = linear_sum_assignment(cost)
            # Note that row_ind will be equal to [0, 1, ..., cost.shape[0]] because cost is a square matrix.
            conf_mat = (-1) * cost[:, col_ind]
            # Accuracy for best permutation:
            acc_d = np.diag(conf_mat).sum() / conf_mat.sum()
        else:
            acc_d = 0 #FIXME
        return acc_d

  #  @classmethod
    def cal_acc(self, model, loader_tr, loader_val, device, i_w, i_h, max_batches=None):
        """
        :param model:
        :param loader:
        :param device: for final test, GPU can be used
        :param max_batches:
                maximum number of iteration for data loader, used to
                probe performance with less computation burden.
                default None, which means to traverse the whole dataset
        """
        model.eval()
        if max_batches is None:
            max_batches = len(loader_tr)
        prediction = Prediction(model, device, loader_tr, loader_val, i_h,i_w, 2)
        _, z_proj, predictions, vec_y_labels, vec_d_labels, image_id_labels = prediction.mk_prediction()
        vec_y_has_strings = any(isinstance(element, str) for element in vec_y_labels)
        vec_d_has_strings = any(isinstance(element, str) for element in vec_d_labels)
        if vec_y_has_strings:
            vec_y_labels = self.encode_array(vec_y_labels)
        if vec_d_has_strings:
            vec_d_labels = self.encode_array(vec_d_labels)
        breakpoint()
        sim_vec_y = self.hungarian_algorithm(predictions, vec_y_labels)
        sim_vec_d = self.hungarian_algorithm(predictions, vec_d_labels)

        return sim_vec_y, sim_vec_d