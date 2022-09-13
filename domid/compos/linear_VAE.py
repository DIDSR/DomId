import torch
from domid.compos.VAE_blocks import linear_block
import torch.nn as nn
import numpy as np

class LinearEncoder(nn.Module):
    def __init__(self, zd_dim, input_dim=(1, 28, 28), features_dim=[500, 500, 2000]):
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

        if x.shape[1]==3:
            x = torch.reshape(x, (x.shape[0], 3*x.shape[2]*x.shape[3]))


        else:
            x = torch.reshape(x, (x.shape[0], 3, x.shape[2]*x.shape[3]))
            x = x[:, 0, :]  # use only the first channel

        z = self.encod(x)
        mu = self.mu_layer(z)
        log_sigma2 = self.log_sigma2_layer(z)
        return mu, log_sigma2


# class LinearDecoder(nn.Module):
#     def __init__(self, zd_dim, input_dim=(28, 28), features_dim=[500, 500, 2000]):
#         """
#         VAE Decoder
#         :param zd_dim: dimension of the latent space
#         :param input_dim: dimension of the oritinal input / output reconstruction, e.g., (28, 28) for MNIST
#         :param features_dim: list of dimensions of the hidden layers
#         """
#         super(LinearDecoder, self).__init__()
#         self.input_dim = input_dim
#         self.decod = nn.Sequential(
#             *linear_block(zd_dim, features_dim[2]),
#             *linear_block(features_dim[2], features_dim[1]),
#             *linear_block(features_dim[1], features_dim[0]),
#             nn.Linear(features_dim[0], np.prod(input_dim)),
#             nn.Sigmoid()
#         )
#         self.decod_2 = nn.Sequential(
#             *linear_block(zd_dim, features_dim[2]),
#             *linear_block(features_dim[2], features_dim[1]),
#             *linear_block(features_dim[1], np.prod(input_dim))
#         )
#
#         self.mu_layer = nn.Linear(np.prod(input_dim), input_dim[0])#input_dim[0], input_dim[1])#shape of the [ 3, 28, 28] FIXME
#         self.log_sigma2_layer = nn.Linear(np.prod(input_dim),input_dim[0])#input_dim[0], input_dim[1])
#     def forward(self, z):
#         """
#         :param z: latent space representation
#         :return x_pro: reconstructed data, which is assumed to have 3 channels, but the channels are assumed to be equal to each other.
#         """
#         #print('here1')
#         x_pro = self.decod(z)
#         #print(x_pro.shape)
#
#         x_pro2 = self.decod_2(z)
#
#         mu = self.mu_layer(x_pro2)
#         sigma = self.log_sigma2_layer(x_pro2)
#
#         #FIXME
#         x_pro = torch.reshape(x_pro, (x_pro.shape[0], 1, *self.input_dim))
#         x_pro = torch.cat((x_pro, x_pro, x_pro), 1)
#
#         x_pro2 = torch.reshape(x_pro2, (x_pro2.shape[0], 1, *self.input_dim)) #FIXME reshape instead of concat
#         x_pro2 = torch.cat((x_pro2, x_pro2, x_pro2), 1)
#
#         return x_pro, x_pro2, mu, sigma

class LinearDecoder(nn.Module):
    def __init__(self, zd_dim, input_dim=(1, 28, 28), features_dim=[500, 500, 2000]):
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
        self.log_sigma_layer = nn.Linear(np.prod(input_dim), np.prod(input_dim))

    def forward(self, z):
        """
        :param z: latent space representation
        :return x_pro: reconstructed data, which is assumed to have 3 channels, but the channels are assumed to be equal to each other.
        """

        x_pro = self.decod(z)

        log_sigma = self.log_sigma_layer(x_pro)
        log_sigma = torch.reshape(log_sigma, (log_sigma.shape[0], *self.input_dim))
        #log_sigma = torch.cat((log_sigma, log_sigma, log_sigma), 1)

        x_pro = torch.reshape(x_pro, (x_pro.shape[0], *self.input_dim))
        #x_pro = torch.cat((x_pro, x_pro, x_pro), 1)
        return x_pro, log_sigma

