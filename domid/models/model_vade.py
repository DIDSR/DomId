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

def mk_vade(parent_class=AModelCluster):
    class ModelVaDE(parent_class):
        def __init__(self, zd_dim, d_dim, device, L, i_c, i_h, i_w, args):
            """
            VaDE model (Jiang et al. 2017 "Variational Deep Embedding:
            An Unsupervised and Generative Approach to Clustering") with
            fully connected encoder and decoder.

            :param zd_dim: dimension of the latent space
            :param d_dim: number of clusters for the clustering task
            :param device: device to use, e.g., "cuda" or "cpu"
            :param i_c: number of channels of the input image
            :param i_h: height of the input image
            :param i_w: width of the input image
            :param args: command line arguments
            """
            super(ModelVaDE, self).__init__()
            self.zd_dim = zd_dim
            self.d_dim = d_dim
            self.device = device
            self.L = L
            self.args = args
            self.loss_epoch = 0

            self.dim_inject_y = 0

            if self.args.dim_inject_y:
                self.dim_inject_y = self.args.dim_inject_y

            # self.dim_inject_domain = 0
            # if self.args.path_to_domain:    # FIXME: one can simply read from the file to find out the injected dimension
            #     self.dim_inject_domain = args.d_dim   # FIXME: allow arbitrary domain vector to be injected

            if self.args.model_method == "linear":
                self.encoder = LinearEncoder(zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
                self.decoder = LinearDecoder(prior=args.prior, zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
                if self.dim_inject_y:
                    warnings.warn("linear model decoder does not support label injection")
            else:
                self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, num_channels=i_c, i_w=i_w, i_h=i_h).to(device)
                self.decoder = ConvolutionalDecoder(
                    prior=args.prior,
                    zd_dim=zd_dim,  # 50
                    domain_dim=self.dim_inject_y,  #
                    # domain_dim=self.dim_inject_y,
                    h_dim=self.encoder.h_dim,
                    num_channels=i_c,
                ).to(device)
            print(self.encoder)
            print(self.decoder)
            self.log_pi = nn.Parameter(
                torch.FloatTensor(
                    self.d_dim,
                )
                .fill_(1.0 / self.d_dim)
                .log(),
                requires_grad=True,
            )
            self.mu_c = nn.Parameter(torch.FloatTensor(self.d_dim, self.zd_dim).fill_(0), requires_grad=True)
            self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.d_dim, self.zd_dim).fill_(0), requires_grad=True)

            #self.loss_writter = SummaryWriter()

        def _inference(self, x):
            """Auxiliary function for inference

            :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
            :return tensor preds_c: One hot encoded tensor of the predicted cluster assignment (shape: [batch_size, self.d_dim]).
            :return tensor probs_c: Tensor of the predicted cluster probabilities; this is q(c|x) per eq. (16) or gamma_c in eq. (12) (shape: [batch_size, self.d_dim]).
            :return tensor z: Tensor of the latent space representation (shape: [batch_size, self.zd_dim])
            :return tensor z_mu: Tensor of the mean of the latent space representation (shape: [batch_size, self.zd_dim])
            :return tensor z_sigma2_log: Tensor of the log of the variance of the latent space representation (shape: [batch_size, self.zd_dim])
            :return tensor mu_c: Tensor of the estimated cluster means (shape: [self.d_dim, self.zd_dim])
            :return tensor log_sigma2_c: Tensor of the estimated cluster variances (shape: [self.d_dim, self.zd_dim])
            :return tensor pi: Tensor of the estimated cluster prevalences, p(c) (shape: [self.d_dim])
            :return tensor logits: Tensor where each column contains the log-probability p(c)p(z|c) for cluster c=0,...,self.d_dim-1 (shape: [batch_size, self.d_dim]).
            """

            z_mu, z_sigma2_log = self.encoder(x)
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
            pi = F.softmax(self.log_pi, dim=0)
            # if torch.any(pi<0.01):
            #     for i in range(len(pi)):
            #         if pi[i]<0.01:
            #
            #             difference = (0.01-pi[i])/(self.d_dim)
            #             pi+= difference
            #             pi[i]=0.01
            # pi[:i]+=difference
            # pi[i+1:]+=difference

            mu_c = self.mu_c
            log_sigma2_c = self.log_sigma2_c

            logits = torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
            # shape [batch_size, self.d_dim], each column contains the log-probability p(c)p(z|c) for cluster c=0,...,self.d_dim-1.

            preds_c, probs_c, *_ = logit2preds_vpic(logits)

            return preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits


        def infer_d_v_2(self, x, inject_domain):
            """
            Used for tensorboard visualizations only.
            """
            results = self._inference(x)
            if len(inject_domain) > 0:
                zy = torch.cat((results[2], inject_domain), 1)
            else:
                zy = results[2]

            # print(results[2].shape, inject_domain.shape, zy.shape)
            x_pro, *_ = self.decoder(zy)
            preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)
            return preds, z_mu, z, log_sigma2_c, probs, x_pro

        def cal_loss(self, x, inject_domain, warmup_beta):
            """Function that is called in trainer_vade to calculate loss

            :param x: tensor with input data
            :return: ELBO loss
            """

            return self._cal_ELBO_loss(x, inject_domain, warmup_beta)



        def _cal_reconstruction_loss(self, x, inject_tensor=[]):
            z_mu = self.encoder.get_z(x)
            z_sigma2_log = self.encoder.get_log_sigma2(x)
            z = z_mu
            if len(inject_tensor) > 0:
                zy = torch.cat((z, inject_tensor), 1)
            else:
                zy = z


            x_pro, *_ = self.decoder(zy)

            if self.args.prior == "Bern":
                L_rec = F.binary_cross_entropy(x_pro, x)
            else:

                sigma = torch.Tensor([0.9]).to(self.device)  # mean sigma of all images
                log_sigma_est = torch.log(sigma).to(self.device)
                L_rec = torch.mean(torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2, 2), 2), 1), 0) / sigma**2

            return L_rec
        def _cal_reconstruction_loss_helper(self, x, x_pro, log_sigma):

            if self.args.prior == "Bern":
                L_rec = F.binary_cross_entropy(x_pro, x)
            else:

                sigma = torch.Tensor([0.9]).to(self.device)  # mean sigma of all images
                log_sigma_est = torch.log(sigma).to(self.device)
                L_rec = torch.mean(torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2, 2), 2), 1), 0) / sigma**2

            return L_rec

        def _cal_ELBO_loss(self, x, inject_domain, warmup_beta):
            """ELBO loss function
            Using SGVB estimator and the reparametrization trick calculates ELBO loss.
            Calculates loss between encoded input and input using ELBO equation (12) in the paper.
            :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
            :param int L: Number of Monte Carlo samples in the SGVB
            """
            preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self._inference(x)
            # mu, sigma from the decoder
            eps = 1e-10

            L_rec = 0.0
            for l in range(self.L):
                z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu  # shape [batch_size, self.zd_dim]4
                if len(inject_domain) > 0:
                    zy = torch.cat((z, inject_domain), 1)
                else:
                    zy = z

                x_pro, log_sigma = self.decoder(zy)  # x_pro, mu, sigma
                L_rec += self._cal_reconstruction_loss_helper(x, x_pro, log_sigma) #FIXME

            L_rec /= self.L
            Loss = L_rec * x.size(1)
            # --> this is the -"first line" of eq (12) in the paper with additional averaging over the batch.

            Loss += (
                0.5
                * warmup_beta
                * torch.mean(
                    torch.sum(
                        probs
                        * torch.sum(
                            log_sigma2_c.unsqueeze(0)
                            + torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0))
                            + (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)),
                            2,
                        ),
                        1,
                    )
                )
            )
            # inner sum dimentions:
            # [1, d_dim, zd_dim] + exp([batch_size, 1, zd_dim] - [1, d_dim, zd_dim]) + ([batch_size, 1, zd_dim] - [1, d_dim, zd_dim])^2 / exp([1, d_dim, zd_dim])
            # = [batch_size, d_dim, zd_dim] -> sum of zd_dim dimensions
            # the next sum is over d_dim dimensions
            # the mean is over the batch
            # --> overall, this is -"second line of eq. (12)" with additional mean over the batch

            Loss -= warmup_beta * torch.mean(
                torch.sum(probs * torch.log(pi.unsqueeze(0) / (probs + eps)), 1)
            )  # FIXME: (+eps) is a hack to avoid NaN. Is there a better way?
            # dimensions: [batch_size, d_dim] * log([1, d_dim] / [batch_size, d_dim]), where the sum is over d_dim dimensions --> [batch_size] --> mean over the batch --> a scalar

            Loss -= 0.5 * warmup_beta * torch.mean(torch.sum(1.0 + z_sigma2_log, 1))
            # dimensions: mean( sum( [batch_size, zd_dim], 1 ) ) where the sum is over zd_dim dimensions and mean over the batch
            # --> overall, this is -"third line of eq. (12)" with additional mean over the batch

            return Loss

        def gaussian_pdfs_log(self, x, mus, log_sigma2s):
            """helper function"""
            loglik = []
            for c in range(self.d_dim):
                loglik.append(self.gaussian_pdf_log(x, mus[c, :], log_sigma2s[c, :]).view(-1, 1))
            return torch.cat(loglik, 1)

        @staticmethod
        def gaussian_pdf_log(x, mu, log_sigma2):
            """
            subhelper function just one gausian pdf log calculation, used as a basis for gaussian_pdfs_log function
            :param x: tensor of shape [batch_size, self.zd_dim]
            :param mu: mean for the cluster distribution
            :param log_sigma2: variance parameters of the cluster distribtion
            :return: tensor with the Gaussian log probabilities of the shape of [batch_size, 1]
            """
            return -0.5 * (
                torch.sum(
                    np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2),
                    1,
                )
            )

    return ModelVaDE
def test_fun(d_dim, zd_dim, device):
    device = torch.device("cpu")
    model = ModelVaDE(d_dim=d_dim, zd_dim=zd_dim, device=device)
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
