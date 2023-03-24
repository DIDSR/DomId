import warnings
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.utils.utils_classif import logit2preds_vpic
from tensorboardX import SummaryWriter
from domainlab.models.a_model_classif import AModelClassif
from domid.compos.cnn_VAE import ConvolutionalDecoder, ConvolutionalEncoder
from domid.compos.linear_VAE import LinearDecoder, LinearEncoder

from domainlab.utils.perf import PerfClassif
from domainlab.utils.perf_metrics import PerfMetricClassif
from domid.utils.perf_cluster import PerfCluster
from rich import print as rprint
import pandas as pd


class ModelVaDE(nn.Module):
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

        self.dim_inject_domain = 0
        if self.args.path_to_domain:    # FIXME: one can simply read from the file to find out the injected dimension
            self.dim_inject_domain = args.d_dim   # FIXME: allow arbitrary domain vector to be injected


        if self.args.model == "linear":
            self.encoder = LinearEncoder(zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
            self.decoder = LinearDecoder(prior=args.prior, zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
            if self.dim_inject_domain or self.dim_inject_y:
                warnings.warn("linear model decoder does not support label injection")
        else:
            self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, num_channels=i_c, i_w=i_w, i_h=i_h).to(device)
            self.decoder = ConvolutionalDecoder(
                prior=args.prior,
                zd_dim=zd_dim, #50
                domain_dim=self.dim_inject_y, #
                #domain_dim=self.dim_inject_y,
                h_dim=self.encoder.h_dim,
                num_channels=i_c
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

        self.loss_writter = tensorboardX.SummaryWriter()

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
        if len(inject_domain)>0:
            zy = torch.cat((results[2], inject_domain), 1)
        else:
            zy = results[2]

        #print(results[2].shape, inject_domain.shape, zy.shape)
        x_pro, *_ = self.decoder(zy)
        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)
        return preds, z_mu, z, log_sigma2_c, probs, x_pro

    def cal_loss(self, x, inject_domain, warmup_beta):
        """Function that is called in trainer_vade to calculate loss

        :param x: tensor with input data
        :return: ELBO loss
        """
        return self.ELBO_Loss(x, inject_domain, warmup_beta)

    def pretrain_loss(self, x, inject_domain):
    
        Loss = nn.MSELoss()
        #Loss = nn.MSELoss(reduction='sum')
        Loss = nn.HuberLoss()
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        if len(inject_domain)>0:
            zy = torch.cat((z, inject_domain), 1)
        else:
            zy = z

        x_pro, *_ = self.decoder(zy)
        
        loss = Loss(x, x_pro)
        return loss

    def reconstruction_loss(self, x, x_pro, log_sigma):

        if self.args.prior == "Bern":
            L_rec = F.binary_cross_entropy(x_pro, x)
        else:
           #  print('first part',torch.mean(torch.sum(torch.sum(torch.sum(log_sigma, 2), 2), 1), 0))
           # #torch.sum(log_sigma)*1/log_sigma.shape[0])
           #  print('second part',  torch.mean(
           #       torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2 / torch.exp(log_sigma) ** 2, 2),2),1), 0))
           #  print('MSE', F.mse_loss(x, x_pro))
           #  print('constant infront of MSE',  torch.sum(torch.exp(log_sigma)**2))
           #  print('x pro min/max/mean', torch.min(x_pro), torch.max(x_pro), torch.mean(x_pro))
           #  print('log_sigma min/max/mean',torch.min(log_sigma), 'max',torch.max(log_sigma), 'mean', torch.mean(log_sigma))

            # L_rec = torch.mean(torch.sum(torch.sum(torch.sum(log_sigma, 2), 2), 1), 0) + torch.mean(
            #     torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2 / torch.exp(log_sigma) ** 2, 2), 2), 1), 0
            # )

            sigma = torch.Tensor([0.9]).to(self.device) #mean sigma of all images
            log_sigma_est = torch.log(sigma).to(self.device)
            L_rec = torch.mean(
                torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2 , 2), 2), 1), 0
            )/sigma**2
            #L_rec = F.mse_loss(x_pro, x)
            # print(L_rec, L_rec0)


            # print('L rec', L_rec)
            # print("#"*10)
            # L_rec = torch.sum(log_sigma)*1/log_sigma.shape[0]+F.mse_loss(x, x_pro)*(log_sigma.shape[0]/torch.sum(torch.exp(log_sigma)**2))

            #L_rec = F.mse_loss(x, x_pro)#*(log_sigma.shape[0]/torch.sum(torch.exp(log_sigma)**2))

        # Note that the mean is taken over the batch dimension, and the sum over the spatial dimensions and the channels.
        # Thir is consistent with the computation of other terms of the ELBO loss below.


        return L_rec

    def ELBO_Loss(self, x, inject_domain, warmup_beta):
        """ELBO loss function
        Using SGVB estimator and the reparametrization trick calculates ELBO loss.
        Calculates loss between encoded input and input using ELBO equation (12) in the papaer.
        :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
        :param int L: Number of Monte Carlo samples in the SGVB
        """
        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self._inference(x)
        # mu, sigma from the decoder
        eps = 1e-10

        L_rec = 0.0
        for l in range(self.L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu  # shape [batch_size, self.zd_dim]4
            if len(inject_domain)>0:
                zy = torch.cat((z, inject_domain), 1)
            else:
                zy = z

            x_pro, log_sigma = self.decoder(zy)  # x_pro, mu, sigma
            L_rec += self.reconstruction_loss(x, x_pro, log_sigma)


        L_rec /= self.L
        Loss = L_rec * x.size(1)
        # --> this is the -"first line" of eq (12) in the paper with additional averaging over the batch.

        Loss += 0.5 *warmup_beta* torch.mean(
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
        # inner sum dimentions:
        # [1, d_dim, zd_dim] + exp([batch_size, 1, zd_dim] - [1, d_dim, zd_dim]) + ([batch_size, 1, zd_dim] - [1, d_dim, zd_dim])^2 / exp([1, d_dim, zd_dim])
        # = [batch_size, d_dim, zd_dim] -> sum of zd_dim dimensions
        # the next sum is over d_dim dimensions
        # the mean is over the batch
        # --> overall, this is -"second line of eq. (12)" with additional mean over the batch

        Loss -= warmup_beta*torch.mean(
            torch.sum(probs * torch.log(pi.unsqueeze(0) / (probs + eps)), 1)
        )  # FIXME: (+eps) is a hack to avoid NaN. Is there a better way?
        # dimensions: [batch_size, d_dim] * log([1, d_dim] / [batch_size, d_dim]), where the sum is over d_dim dimensions --> [batch_size] --> mean over the batch --> a scalar

        Loss -= 0.5 * warmup_beta*torch.mean(torch.sum(1.0 + z_sigma2_log, 1))
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
    def create_perf_obj(self, task):
        """
        for classification, dimension of target can be quieried from task
        """

        self.perf_metric = PerfCluster(task.dim_y) #PerfMetricClassif(task.dim_y)
        return self.perf_metric
    def cal_perf_metric(self, loader_tr, device, loader_te=None):
        """
        classification performance matric
        """

        metric_te = None
        metric_tr = None
        with torch.no_grad():
            metric_te = self.perf_metric.cal_acc(self, loader_te, device)
            metric_tr = self.perf_metric.cal_acc(self, loader_tr, device)
            # metric_tr_pool = self.perf_metric.cal_metrics(self, loader_tr, device)
            # confmat = metric_tr_pool.pop("confmat")
            # print("pooled train domains performance:")
            # print(metric_tr_pool)
            # print("confusion matrix:")
            # print(pd.DataFrame(confmat))
            # metric_tr_pool["confmat"] = confmat
            # # test set has no domain label, so can be more custom
            # if loader_te is not None:
            #     metric_te = self.perf_metric.cal_metrics(self, loader_te, device)
            #     confmat = metric_te.pop("confmat")
            #     print("out of domain test performance:")
            #     print(metric_te)
            #     print("confusion matrix:")
            #     print(pd.DataFrame(confmat))
            #     metric_te["confmat"] = confmat

        return metric_tr


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
