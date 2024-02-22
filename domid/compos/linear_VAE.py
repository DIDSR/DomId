import numpy as np
import torch
import torch.nn as nn

from domid.compos.VAE_blocks import linear_block


class LinearEncoder(nn.Module):
    def __init__(self, zd_dim, input_dim=(3, 28, 28), features_dim=[500, 500, 2000]):
        """
        VAE Encoder with linear layers

        :param zd_dim: dimension of the latent space
        :param input_dim: dimensions of the input, e.g., (3, 28, 28) for MNIST in RGB format
        :param features_dim: list of dimensions of the hidden layers
        """
        super(LinearEncoder, self).__init__()
        self.input_dim = np.prod(input_dim)
        self.encod = nn.Sequential(
            *linear_block(self.input_dim, features_dim[0]),
            *linear_block(features_dim[0], features_dim[1]),
            *linear_block(features_dim[1], features_dim[2]),
        )
        self.mu_layer = nn.Linear(features_dim[2], zd_dim)
        self.log_sigma2_layer = nn.Linear(features_dim[2], zd_dim)
    #FIXME def get_z(self, x):
    # mu, log_sigma2 = self.forward(x)
    # return mu

    def get_z(self, x):
        mu, log_sigma2 = self.forward(x)
        return mu

    def get_log_sigma2(self, x):
        mu, log_sigma2 = self.forward(x)
        return log_sigma2
    def forward(self, x):
        """
        :param x: input data, assumed to have 3 channels
        """

        assert x.shape[1] == 3
        x = torch.reshape(x, (x.shape[0], 3 * x.shape[2] * x.shape[3]))
        z = self.encod(x)
        mu = self.mu_layer(z)
        log_sigma2 = self.log_sigma2_layer(z)
        return mu, log_sigma2


class LinearDecoder(nn.Module):
    def __init__(self, prior, zd_dim, input_dim=(3, 28, 28), features_dim=[500, 500, 2000]):
        """
        VAE Decoder

        :param zd_dim: dimension of the latent space
        :param input_dim: dimension of the original input / output reconstruction, e.g., (3, 28, 28) for MNIST in RGB format
        :param features_dim: list of dimensions of the hidden layers, given in reverse order
        """
        self.prior = prior
        super(LinearDecoder, self).__init__()
        self.input_dim = input_dim
        self.decod = nn.Sequential(
            *linear_block(zd_dim, features_dim[2]),
            *linear_block(features_dim[2], features_dim[1]),
            *linear_block(features_dim[1], features_dim[0]),
        )
        self.mu_layer = nn.Linear(features_dim[0], np.prod(input_dim))
        self.log_sigma_layer = nn.Linear(features_dim[0], np.prod(input_dim))
        self.activation = nn.Sigmoid()

    def forward(self, z):
        """
        :param z: latent space representation
        :return x_pro: reconstructed data, which is assumed to have 3 channels, but the channels are assumed to be equal to each other.
        """
        x_decoded = self.decod(z)

        if self.prior == "Bern":
            # if Bernoulli distribution sigmoid activation to mu is applied
            x_pro = self.mu_layer(x_decoded)
            x_pro = self.activation(x_pro)
        else:
            x_pro = self.mu_layer(x_decoded)

        log_sigma = self.log_sigma_layer(x_decoded)

        x_pro = torch.reshape(x_pro, (x_pro.shape[0], *self.input_dim))
        log_sigma = torch.reshape(log_sigma, (log_sigma.shape[0], *self.input_dim))

        return x_pro, log_sigma
