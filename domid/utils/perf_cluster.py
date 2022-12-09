# FIXME: clean up the following discussion...
# rows are ground truth cluster label
# columns are the predicted cluster label
# entries are number of instances
#
# one-hot encoded clustering output for the first five data points:
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [1., 0., 0.],
#        [0., 1., 0.],
#        [0., 1., 0.]], dtype=float32)
# 
# ground truth for the first five data points:
# In [43]: out["cluster_true"][0][:5]
# Out[43]: array([0, 1, 2, 0, 1])
#
# 1. look at the first entry in the array
#|    | t1 | t2 | t3 |
#|----+----+----+----|
#| p1 | +1 |    |    |
#| p2 |    |    |    |
#| p3 |    |    |    |
#|----+----+----+----|
#
# 2. look at the second entry in the array
#|    | t1 | t2 | t3 |
#|----+----+----+----|
#| p1 | 1  |    |    |
#| p2 |    | +1 |    |
#| p3 |    |    |    |
#|----+----+----+----|
#
# 3. look at the third entry in the array
#|    | t1 | t2 | t3 |
#|----+----+----+----|
#| p1 | 1  |    | +1 |
#| p2 |    |  1 |    |
#| p3 |    |    |    |
#|----+----+----+----|
#
# 4. look at the 4th and 5th entry in the array
#|    | t1 | t2 | t3 |
#|----+----+----+----|
#| p1 | 1  |  0 |  1 |
#| p2 | +1 |1+1 |  0 |
#| p3 | 0  | 0  |  0 |
#|----+----+----+----|
#
## What is the best permutation?
# we are trying to maximize the sum of the diagonal entries.
#cost = - np.array([[1, 0, 1], [1, 2, 0], [0, 0, 0]])
#row_ind, col_ind = linear_sum_assignment(cost)
#col_ind
#Out[52]: array([0, 1, 2])
#cost[row_ind, col_ind].sum()  # note that this is not summing the full matrix but only the entries defined by zip(row_ind, col_ind)
#Out[53]: -3
#
# # simpler way to construct the same matrix:
# tmp_pred = np.array([[1., 0., 0.],
#                      [0., 1., 0.],
#                      [1., 0., 0.],
#                      [0., 1., 0.],
#                      [0., 1., 0.]])
# tmp_true = np.array([0, 1, 2, 0, 1])
# confusion_matrix(tmp_pred.argmax(axis=1), tmp_true)
# # Out[]: array([[1, 0, 1],
# #               [1, 2, 0],
# #               [0, 0, 0]])
# # 
#
# list(itertools.permutations([1, 2, 3]))
# >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
# >>> from scipy.optimize import linear_sum_assignment
# >>> row_ind, col_ind = linear_sum_assignment(cost)
# >>> col_ind
# >>> cost[row_ind, col_ind].sum()


import numpy as np
import torch
from domainlab.utils.perf import PerfClassif
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


class PerfCluster(PerfClassif):
    """Clustering Performance"""

    @classmethod
    def cal_acc(cls, model, loader_te, device, max_batches=None):
        """
        :param model:
        :param loader_te:
        :param device: for final test, GPU can be used
        :param max_batches:
                maximum number of iteration for data loader, used to
                probe performance with less computation burden.
                default None, which means to traverse the whole dataset
        """
        model.eval()
        model_local = model.to(device)
        if max_batches is None:
            max_batches = len(loader_te)
        list_vec_preds, list_vec_labels = [], []
        cost = np.zeros((model_local.d_dim, model_local.d_dim), dtype="int")
        with torch.no_grad():
            for i, (x_s, _, d_s, *_) in enumerate(loader_te):
                x_s, d_s = x_s.to(device), d_s.to(device)
                pred = model_local.infer_d_v(x_s)
                # number of predicted clusters can be larger than the number of ground truth clusters
                assert pred.shape >= d_s.shape
                # there are d_dim possible predicted clusters
                assert pred.shape[1] == model_local.d_dim

                cluster_pred_scalar = pred.cpu().numpy().argmax(axis=1)
                cluster_true_scalar = d_s.cpu().numpy().argmax(axis=1)
                cost = cost - confusion_matrix(cluster_pred_scalar, cluster_true_scalar,
                                               labels=list(range(model_local.d_dim)))

                list_vec_preds.append(pred)
                list_vec_labels.append(d_s)
                if i > max_batches:
                    break
                
        # The domain label are never used in training. so we need to find
        # correspondence between predicted and true domain indices. See top of
        # this file for an explanation.

        # What is the best permutation?
        row_ind, col_ind = linear_sum_assignment(cost)
        # Note that row_ind will be equal to [0, 1, ..., cost.shape[0]] because cost is a square matrix.
        conf_mat = (-1)*cost[:, col_ind]
        # Accuracy for best permutation:
        acc_d = np.diag(conf_mat).sum() / conf_mat.sum()

        return acc_d, conf_mat
