import torch
import torch.nn as nn
import torch.nn.functional as F
from domid.utils.perf_cluster import PerfCluster

class AModelCluster(nn.Module):
    """
    Operations that all clustering models should have
    """
    def __init__(self):
        super(AModelCluster, self).__init__()
        self._decoratee = None

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

    def extend(self, model):
        """
        extend the loss of the decoratee
        """
        self._decoratee = model
        self.reset_feature_extractor(model.net_invar_feat)
    def cal_loss(self, tensor_x, vec_y, vec_d, inj_tensor, img_ids):
        """
        Calculates the loss for the model.
        """

        raise NotImplementedError

    def infer_d_v(self, x):
        """
        Predict the cluster/domain of the input data.
        Corresponds to equation (16) in the paper.

        :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
        :return tensor preds: One hot encoded tensor of the predicted cluster assignment.
        """
        preds, *_ = self._inference(x)
        return preds.cpu().detach()

    def _extend_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        combine losses from two models
        """
        if self._decoratee is not None:
            return self._decoratee.cal_rec_loss(
                tensor_x, tensor_y, tensor_d, others)
        return None, None
    def _cal_pretrain_loss(self, tensor_x, x_pro, inject_domain=None):
        """
        Pretraining loss for the model.
        """
        Loss = nn.MSELoss()
        pre_loss = Loss(tensor_x, x_pro)
        raise pre_loss
    def _cal_reconstruction_loss(self, x, x_pro):
        """
        Reconstruction loss for the model.
        """
        sigma = torch.Tensor([0.9]).to(self.device)  # mean sigma of all images
        log_sigma_est = torch.log(sigma).to(self.device)
        rec_loss = torch.mean(torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2, 2), 2), 1), 0) / sigma ** 2

        raise rec_loss
    def _cal_kl_loss(self, q, p):
        """

        :return:
        """
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        return kl_loss
