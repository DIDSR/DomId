import torch
import torch.nn as nn

from domid.utils.perf_cluster import PerfCluster


class AModelCluster(nn.Module):
    """
    Operations that all clustering models should have
    """

    def create_perf_obj(self, task):
        """
        Sets up the performance metrics used.
        """
        self.perf_metric = PerfCluster(task.dim_y)
        return self.perf_metric

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

        return metric_tr, metric_te
