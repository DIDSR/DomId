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
        self.task = task
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

        r_score_tr = None
        r_score_te = None
        # if self.task.get_list_domains() == ['class0', 'class1', 'class2']: #if task ==her2
        if hasattr(self.task, "calc_corr"):
            with torch.no_grad():
                r_score_tr, r_score_te = self.task.calc_corr(self, loader_tr, loader_te, device)

        return metric_tr, metric_te, r_score_tr, r_score_te
