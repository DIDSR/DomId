import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.utils.utils_classif import logit2preds_vpic
from tensorboardX import SummaryWriter

from domid.compos.cnn_VAE import ConvolutionalDecoder, ConvolutionalEncoder
from domid.compos.linear_VAE import LinearDecoder, LinearEncoder
from domid.models.a_model_cluster import AModelCluster
from domid.compos.GNN_layer import GNNLayer
from domid.compos.linear_AE import LinearEncoderAE, LinearDecoderAE
from domid.compos.GNN import GNN

import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from domainlab.dsets.utils_data import mk_fun_label2onehot
from datetime import datetime
class ModelAE(AModelCluster):
    def __init__(self, zd_dim, d_dim, device, L, i_c, i_h, i_w, args):

        super(ModelAE, self).__init__()
        self.zd_dim = zd_dim
        self.d_dim = d_dim
        self.device = device
        self.L = L
        self.args = args
        self.loss_epoch = 0

        self.dim_inject_y = 0

        if self.args.dim_inject_y:
            self.dim_inject_y = self.args.dim_inject_y

        n_clusters = d_dim
        n_z = zd_dim
        n_input = i_c * i_h * i_w
        n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, = 500, 500, 2000, 2000, 500, 500,

        self.cluster_layer = nn.Parameter(torch.Tensor(self.d_dim, self.zd_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


        self.encoder =  LinearEncoderAE(n_enc_1, n_enc_2, n_enc_3,n_input, n_z)
        self.decoder = LinearDecoderAE(n_dec_1, n_dec_2, n_dec_3, n_input, n_z)

        self.counter= 0


    def _inference(self, x, inject_tensor=None):
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        tra1, tra2, tra3, z = self.encoder(x)
        # _, _, z, *_ = self._inference(x)

        kmeans = KMeans(n_clusters=self.args.d_dim, n_init=20)

        kmeans.fit_predict(z.detach().cpu().numpy())
        x_bar, *_ = self.decoder(z)
        z_mu = torch.mean(z, dim=0)
        z_sigma2_log = torch.std(z, dim=0)
        pi = torch.Tensor([0])
        predictions = kmeans.labels_
        preds_c = mk_fun_label2onehot(10)(predictions)
        logits = torch.Tensor(kmeans.fit_transform(z.detach().cpu().numpy())).to(self.device)
        _,probs_c, *_ = logit2preds_vpic(logits)
        # preds_c: One hot encoded tensor of the predicted cluster assignment (shape: [batch_size, self.d_dim]).
        # probs_c: Tensor of the predicted cluster probabilities; this is q(c|x) per eq. (16) or gamma_c in eq. (12) (shape: [batch_size, self.d_dim]).
        # logits: Tensor where each column contains the log-probability p(c)p(z|c) for cluster c=0,...,self.d_dim-1 (shape: [batch_size, self.d_dim])
        return preds_c, probs_c, z, z_mu, z_sigma2_log, x_bar, z_sigma2_log, pi, logits

    def infer_d_v(self, x):
        """
        Predict the cluster/domain of the input data.
        Corresponds to equation (16) in the paper.

        :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
        :return tensor preds: One hot encoded tensor of the predicted cluster assignment.
        """
        preds, *_ = self._inference(x)
        return preds.cpu().detach()

    def infer_d_v_2(self, x, inject_domain):
        """
        Used for tensorboard visualizations only.
        """
        results = self._inference(x)
        if len(inject_domain) > 0:
            zy = torch.cat((results[2], inject_domain), 1)
        else:
            zy = results[2]
        x_pro, *_ = self.decoder(zy)


        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)
        return preds, z_mu, z, log_sigma2_c, probs, x_pro

    def cal_loss(self, x, inject_domain, warmup_beta=None):
        loss = self.pretrain_loss(x, inject_domain)
        return loss

    def pretrain_loss(self, x, inject_domain):
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        *_, z = self.encoder(x)

        if len(inject_domain) > 0:
            zy = torch.cat((z, inject_domain), 1)
        else:
            zy = z
        x_pro, *_ = self.decoder(zy) #FIXME account for different number of outputs from decoder

        loss = F.mse_loss(x_pro, x)

        return loss




def test_fun(d_dim, zd_dim, device):
    device = torch.device("cpu")
    model = ModelAE(d_dim=d_dim, zd_dim=zd_dim, device=device)
    x = torch.rand(2, 3, 28, 28)
    import numpy as np

    a = np.zeros((2, 10))
    a = np.double(a)
    a[0, 1] = 1.0
    a[1, 8] = 1.0
    a
    y = torch.tensor(a, dtype=torch.float)
    model(x, y)
    model.cal_loss(x)
