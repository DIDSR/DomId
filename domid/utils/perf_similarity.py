import numpy as np
import torch
from domainlab.utils.perf import PerfClassif
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from domid.compos.predict_basic import Prediction
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

    def hungarian_algorithm(self, arr1, arr2):
        """
        This function takes two arrays as input, encodes any string elements to integers,
        and applies the Hungarian Algorithm to find the optimal assignment between the two arrays.
        """

        cost_matrix = np.zeros((len(arr1), len(arr2)))
        for i in range(len(arr1)):
            for j in range(len(arr2)):
                cost_matrix[i][j] = abs(arr1[i] - arr2[j])
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        optimal_assignments = [(arr1[row_ind[i]], arr2[col_ind[i]]) for i in range(len(row_ind))]
        breakpoint()
        return optimal_assignments

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


        hungarian_assignments = self.hungarian_algorithm(vec_y_labels, vec_d_labels)
        breakpoint()
        sim_vec_d = 0
        sim_vec_y = 0
        return sim_vec_y, sim_vec_d