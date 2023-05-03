import numpy as np
import torch
from domainlab.utils.perf import PerfClassif
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import scipy
import os
import pandas as pd
class PerfCorrelation(PerfClassif):
    """Clustering Performance"""

    def __init__(self):
        super().__init__()

    @classmethod
    def load_csv_for_her2(clc, path):
        """
        This function takes a path to a csv file as input and returns a list of the labels in the file.
        """

        clc.df = pd.read_csv(os.path.join(path, 'dataframe.csv'))

        return clc.df

    @classmethod
    def mean_score_order_match(clc, img_locs, true_scores):

        M = []
        for prediction in img_locs:
          M.append(int(true_scores[true_scores['img_id']==prediction.split('/')[-1]]['score']))

        return M

    @classmethod
    def domain_class_mapping(clc, domain_predictions, true_scores):
        dic1 = {0: 2, 1: 1, 2: 0}
        dic2 = {0: 1, 1: 0, 2: 2}
        dic3 = {0: 2, 1: 0, 2: 1}
        dic4 = {0: 0, 1: 2, 2: 1}
        dic5 = {0: 2, 1: 1, 2: 1}
        dic6 = {0: 0, 1: 2, 2: 0}
        dictionaries = [dic1, dic2, dic3, dic4, dic5, dic6]
        combos = []
        for i in range(0, 6):
            mapping = dictionaries[i]
            new_combination = []
            for j in domain_predictions:
                new_combination.append(mapping[j])
            combos.append(new_combination)

        R_values = []

        for i in combos:
            r = np.corrcoef(i, true_scores)
            R_values.append(r[0][1])
        print(R_values)

        return max(R_values)
    @classmethod
    def cal_acc(clc, model, loader_tr, device, max_batches=None):

        model.eval()
        model_local = model.to(device)
        df = clc.load_csv_for_her2('../../HER2/combined_train') #FIXME
        image_id_labels =[]
        domain_predictions = []
        with torch.no_grad():
            for i, (x_s, y_s, d_s, _, img_id) in enumerate(loader_tr):

                x_s, y_s, d_s = x_s.to(device), y_s.to(device), d_s.to(device)

                pred = model_local.infer_d_v(x_s)
                domain_predictions+=list(pred.argmax(dim=1).cpu().numpy())
                image_id_labels+=img_id
        mean_scores = clc.mean_score_order_match(image_id_labels, df)
        R_scores = clc.domain_class_mapping(domain_predictions, mean_scores)

        return R_scores