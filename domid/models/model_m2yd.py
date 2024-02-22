import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from domainlab.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv import (
    DecoderConcatLatentFCReshapeConvGatedConv,
)

from domainlab.compos.vae.compos.encoder import LSEncoderLinear as LSEncoderDense
from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.utils_class import store_args
from domainlab.utils.utils_classif import get_label_na, logit2preds_vpic

from domid.compos.nn_net import Net_MNIST

def mk_m2yd(parent_class=AModelClassif):
    class ModelXY2D(AModelClassif):
        """
        Let zd to be continuous vector, each component of zd represents the "attention" weight.
        For a cluster, that means the bigger the $zd_k$ value, the more likely the cluster assignment
        to component $k$. Note $zd_k~N(0,1)$.

        Computational Structure:
        generative path: (zd,y) -> x (image)
        generative path: N(0,I) -> zd (prior for zd)
        variational posterior path:
        1. x -> y,
        2. [y, feat(x)] -> z_d

        FIXME: if we change the variational inference order, instead of x->y, then [y,feat(x)]->z_d
        if we do first x->d (style extraction, texture prediction),
        then [feat(x),d]-> y,  will this be better?

        KL divergence between posterior path vs generative(prior) path:
        1. x->y: auxiliary path (not regularized by generative path, but by supervised learning)
        no KL for y
        2. KL(q(z_d)|p(z_d)),
        q(z_d): [y,feat(x)]-> z_d
        p(z_d): N(0,I)

        Auxilliary path: supervised learning of x->y

        FIXME: original M2 has prior Gaussian(0, I) for $z_d$, will this hinder learning of $z_d$
        on each component since the prior is draging each component to zero.
        """


        @store_args
        def __init__(self, list_str_y, y_dim, zd_dim, gamma_y, device, i_c, i_h, i_w, dim_feat_x=10, list_str_d=None):
            """
            :param y_dim: classification task class-label dimension
            :param zd_dim: dimension of latent variable $z_d$ dimension
            :param aux_y:
            """
            super().__init__(list_str_y, list_str_d)
            self.d_dim = zd_dim  # number of domains
            self.infer_y_from_x = Net_MNIST(y_dim, self.i_h)
            self.feat_x2concat_y = Net_MNIST(self.dim_feat_x, self.i_h)
            # FIXME: shall we share parameters between infer_y_from_x and self.feat_x2concat_y?
            self.infer_domain = LSEncoderDense(z_dim=self.zd_dim, dim_input=self.dim_feat_x + self.y_dim)
            self.gamma_y = gamma_y
            # LN: location scale encoder
            self.decoder = DecoderConcatLatentFCReshapeConvGatedConv(
                z_dim=zd_dim + y_dim, i_c=self.i_c, i_w=self.i_w, i_h=self.i_h
            )

        def cal_logit_y(self, tensor_x):
            """
            calculate the logit for softmax classification
            """
            return self.infer_y_from_x(tensor_x)

        def infer_y_vpicn(self, tensor_x):
            """
            :param tensor_x: input tensor
            :return:
                - vec_one_hot - (list) one-hot encoded classification output
                - prob - (list) softmax probabilities per class
                - ind - (int) index of maximal output class score
                - confidence - (float) maximum probability (already included in prob)
                - na_class - (string) class label for the maximum probability class
            """
            with torch.no_grad():
                logit_y = self.infer_y_from_x(tensor_x)
            vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)
            na_class = get_label_na(ind, self.list_str_y)
            return vec_one_hot, prob, ind, confidence, na_class

        def infer_d_v(self, tensor_x):
            with torch.no_grad():
                y_hat_logit = self.infer_y_from_x(tensor_x)
                feat_x = self.feat_x2concat_y(tensor_x)
                feat_y_x = torch.cat((y_hat_logit, feat_x), dim=1)
                q_zd, zd_q = self.infer_domain(feat_y_x)
            vec_one_hot, *_ = logit2preds_vpic(q_zd.mean)
            return vec_one_hot

        def forward(self, tensor_x, vec_y, vec_d=None):
            y_hat_logit = self.infer_y_from_x(tensor_x)
            feat_x = self.feat_x2concat_y(tensor_x)
            feat_y_x = torch.cat((y_hat_logit, feat_x), dim=1)
            q_zd, zd_q = self.infer_domain(feat_y_x)
            return q_zd, zd_q, y_hat_logit

        def cal_loss(self, x, y, d=None, others=None):
            q_zd, zd_q, y_hat = self.forward(x, y)
            z_con = torch.cat((zd_q, y), dim=1)  # FIXME: pay attention to order

            nll, x_mean, x_logvar = self.decoder(z_con, x)
            pzd_loc = torch.zeros(1, self.zd_dim).to(self.device)
            pzd_scale = torch.ones(self.zd_dim).to(self.device)
            pzd = dist.Normal(pzd_loc, pzd_scale)

            pzd_log_prob = pzd.log_prob(zd_q)
            qzd_log_prob = q_zd.log_prob(zd_q)
            zd_p_minus_zd_q = torch.sum(pzd_log_prob - qzd_log_prob, dim=1)
            # FIXME: use analytical expression since using sampling to estimate KL divergence is high variance
            _, y_target = y.max(dim=1)  # y is the observed class label, not the cluster label!
            lc_y = F.cross_entropy(y_hat, y_target, reduction="none")
            loss = nll - zd_p_minus_zd_q + self.gamma_y * lc_y
            return loss.mean()
    return ModelXY2D

def test_fun():
    model = ModelXY2D(y_dim=10, zd_dim=8, gamma_y=3500, device=torch.device("cpu"), i_c=3, i_h=28, i_w=28)
    device = torch.device("cpu")
    x = torch.rand(2, 3, 28, 28)
    import numpy as np

    a = np.zeros((2, 10))
    a = np.double(a)
    a[0, 1] = 1.0
    a[1, 8] = 1.0
    a
    y = torch.tensor(a, dtype=torch.float)
    model(x, y)
    model.cal_loss(x, y)
