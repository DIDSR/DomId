import torch
import torch.nn as nn

from domid.utils.perf_cluster import PerfCluster
from domid.utils.perf_similarity import PerfCorrelation

class AModelCluster(nn.Module):
    """
    Operations that all clustering models should have
    """

    def create_perf_obj(self, task):
        """
        Sets up the performance metrics used.
        """
        self.task = task
        self.perf_metric = PerfCluster(task.dim_y)


        self.perf_metric_correlation = PerfCorrelation()
        return self.perf_metric, self.perf_metric_correlation

    def cal_perf_metric(self, loader_tr, device, loader_te=None):
        """
        Clustering performance metric on the training and test/validation sets.
        """
        metric_te = None
        metric_tr = None
        with torch.no_grad():
            metric_tr = self.perf_metric.cal_acc(self, loader_tr, device)
            if loader_te is not None:
                metric_te = self.perf_metric.cal_acc(self, loader_te, device)

        r_score_tr = None
        if self.task.get_list_domains() == ['class0', 'class1', 'class2']: #if task ==her2
            with torch.no_grad():
                r_score_tr = self.perf_metric_correlation.cal_acc(self, loader_tr, loader_te,device, self.task)

        return metric_tr, metric_te, r_score_tr


