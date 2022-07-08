import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libdg.utils.utils_classif import logit2preds_vpic


def linear_block(in_c, out_c):
    layers = [nn.Linear(in_c, out_c), nn.ReLU(True)]
    return layers


class LinearEncoder(nn.Module):
    def __init__(self, zd_dim, input_dim=(28, 28), features_dim=[300, 500, 700]):
        """
        VAE Encoder
        :param zd_dim: dimension of the latent space
        :param input_dim: dimensions of the input, e.g., (28, 28) for MNIST
        :param features_dim: list of dimensions of the hidden layers
        """
        super(LinearEncoder, self).__init__()
        self.input_dim = np.prod(input_dim)
        self.encod = nn.Sequential(
            *linear_block(self.input_dim, features_dim[0]),
            *linear_block(features_dim[0], features_dim[1]),
            *linear_block(features_dim[1], features_dim[2])
        )
        self.mu_layer = nn.Linear(features_dim[2], zd_dim)
        self.log_sigma2_layer = nn.Linear(features_dim[2], zd_dim)

    def forward(self, x):
        """
        :param x: input data, assumed to have 3 channels, but only the first one is passed through the network.
        """
        x = torch.reshape(x, (x.shape[0], 3, self.input_dim))
        x = x[:, 0, :]  # use only the first channel
        z = self.encod(x)
        mu = self.mu_layer(z)
        log_sigma2 = self.log_sigma2_layer(z)
        return mu, log_sigma2


class LinearDecoder(nn.Module):
    def __init__(self, zd_dim, input_dim=(28, 28), features_dim=[300, 500, 700]):
        """
        VAE Decoder
        :param zd_dim: dimension of the latent space
        :param input_dim: dimension of the oritinal input / output reconstruction, e.g., (28, 28) for MNIST
        :param features_dim: list of dimensions of the hidden layers
        """
        super(LinearDecoder, self).__init__()
        self.input_dim = input_dim
        self.decod = nn.Sequential(
            *linear_block(zd_dim, features_dim[2]),
            *linear_block(features_dim[2], features_dim[1]),
            *linear_block(features_dim[1], features_dim[0]),
            nn.Linear(features_dim[0], np.prod(input_dim)),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        :param z: latent space representation
        :return x_pro: reconstructed data, which is assumed to have 3 channels, but the channels are assumed to be equal to each other.
        """
        x_pro = self.decod(z)
        x_pro = torch.reshape(x_pro, (x_pro.shape[0], 1, *self.input_dim))
        x_pro = torch.cat((x_pro, x_pro, x_pro), 1)
        return x_pro


class ModelVaDE(nn.Module):
    def __init__(self, zd_dim, d_dim, device, i_c, i_h, i_w):
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
        """
        super(ModelVaDE, self).__init__()
        self.zd_dim = zd_dim
        self.d_dim = d_dim
        self.device = device

        self.encoder = LinearEncoder(zd_dim=zd_dim, input_dim=(i_h, i_w)).to(device)
        self.decoder = LinearDecoder(zd_dim=zd_dim, input_dim=(i_h, i_w)).to(device)

        self.pi_ = nn.Parameter(torch.FloatTensor(self.d_dim,).fill_(1) / self.d_dim,
                                requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.d_dim, self.zd_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.d_dim, self.zd_dim).fill_(0), requires_grad=True)

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

        pi = self.pi_
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

    def infer_d_v_2(self, x):
        """
        Used for tensorboard visualizations only.
        """
        results = self._inference(x)
        x_pro = self.decoder(results[2])
        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)
        return preds, z_mu, z, log_sigma2_c, probs, x_pro

    def cal_loss(self, x):
        """Function that is called in trainer_vade to calculate loss
        :param x: tensor with input data
        :return: ELBO loss
        """
        return self.ELBO_Loss(x)

    def ELBO_Loss(self, x, L=1):
        """ELBO loss function

        Using SGVB estimator and the reparametrization trick calculates ELBO loss.
        Calculates loss between encoded input and input using ELBO equation (12) in the papaer.

        :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
        :param int L: Number of Monte Carlo samples in the SOVB
        """
        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self._inference(x)
        eps = 1e-10

        L_rec = 0.0
        for l in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu  # shape [batch_size, self.zd_dim]
            x_pro = self.decoder(z)
            L_rec += F.binary_cross_entropy(
                x_pro, x
            )  # TODO: this is the reconstruction loss for a binary-valued x (such as MNIST digits); need to implement another version for a real-valued x.

        L_rec /= L
        Loss = L_rec * x.size(1)
        # doesn't take the mean over the channels; i.e., the recon loss is taken as an average over (batch size * L * width * height)
        # --> this is the -"first line" of eq (12) in the paper with additional averaging over the batch.

        Loss += 0.5 * torch.mean(
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

        Loss -= torch.mean(torch.sum(probs * torch.log(pi.unsqueeze(0) / (probs)), 1))
        # dimensions: [batch_size, d_dim] * log([1, d_dim] / [batch_size, d_dim]), where the sum is over d_dim dimensions --> [batch_size] --> mean over the batch --> a scalar
        Loss -= 0.5 * torch.mean(torch.sum(1.0 + z_sigma2_log, 1))
        # dimensions: mean( sum( [batch_size, zd_dim], 1 ) ) where the sum is over zd_dim dimensions and mean over the batch
        # --> overall, this is -"third line of eq. (12)" with additional mean over the batch

        return Loss

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        """helper function
        """
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
