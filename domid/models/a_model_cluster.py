import torch
import torch.nn as nn
import torch.nn.functional as F
from domid.utils.perf_cluster import PerfCluster
import abc
class AModelCluster(nn.Module):
    """
    Operations that all clustering models should have
    """
    def __init__(self):
        super(AModelCluster, self).__init__()
        self._decoratee = None #FIXME do i pass it to every model?



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

    def cal_loss(self, tensor_x, vec_y=None, vec_d=None, inj_tensor=[]):
        """
        Calculates the loss for the model.
        """

        total_loss = self._cal_reconstruction_loss(tensor_x, inj_tensor)
        #if self._decoratee is not None:

        kl_loss = self._cal_kl_loss(tensor_x, vec_y, vec_d, inj_tensor)

        total_loss += kl_loss
        return total_loss

    def infer_d_v(self, x):
        """
        Predict the cluster/domain of the input data.
        Corresponds to equation (16) in the paper.

        :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
        :return tensor preds: One hot encoded tensor of the predicted cluster assignment.
        """
        preds, *_ = self._inference(x)
        return preds.cpu().detach()
    def extend(self, model):
        """
        extend the loss of the decoratee
        """
        self._decoratee = model


    def _extend_loss(self, tensor_x, tensor_y, tensor_d, others=None):
        """
        combine losses from two models
        """
        if self._decoratee is not None:
            return self._decoratee._cal_kl_loss(
                tensor_x, tensor_y, tensor_d, others)
        return None, None

    @abc.abstractmethod
    def _cal_pretrain_loss(self, tensor_x, inject_tensor=None):
        """
        Pretraining loss for the model.
        """
        return self._cal_reconstruction_loss(tensor_x, inject_tensor)
    def _cal_reconstruction_loss(self, tensor_x, inject_domain=None):

        if self.args.model == "linear":
            tensor_x = torch.reshape(tensor_x, (tensor_x.shape[0], tensor_x.shape[1] * tensor_x.shape[2] * tensor_x.shape[3]))

        if self.args.feat_extract == "vae":
            z_mu, z_sigma2_log = self.encoder(tensor_x)
            z = z_mu
            if len(inject_domain) > 0:
                zy = torch.cat((z, inject_domain), 1)
            else:
                zy = z
        elif self.args.feat_extract == "ae":
            *_, z_mu = self.encoder(tensor_x)
            zy = z_mu



        x_pro = self.decoder(zy)

        if isinstance(x_pro, tuple):
            x_pro = x_pro[0]

        loss = F.mse_loss(x_pro, tensor_x)

        return loss
    @abc.abstractmethod
    def _cal_kl_loss(self, q, p): #FIXME KL loss is different for each of the model, redefined it in every model?

        return NotImplementedError
