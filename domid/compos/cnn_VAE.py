import torch.nn as nn
from domid.compos.VAE_blocks import get_output_shape, UnFlatten
import numpy as np


class ConvolutionalEncoder(nn.Module):
    def __init__(self, zd_dim, input_dim=3, features_dim=[32, 64, 128], i_w=28, i_h=28):
        """
        VAE Encoder
        :param zd_dim: dimension of the latent space
        :param input_dim: dimensions of the input, e.g., (28, 28) for MNIST
        :param features_dim: list of dimensions of the hidden layers
        """
        super(ConvolutionalEncoder, self).__init__()
        self.input_dim = np.prod(input_dim)
        self.encod = nn.Sequential()
        features_dim = [input_dim] + features_dim
        k = [3, 3, 3]  # , 3, 3, 3]
        for i in range(len(features_dim) - 1):
            self.encod.append(nn.Conv2d(features_dim[i], features_dim[i + 1], kernel_size=k[i], stride=2, padding=1))
            self.encod.append(nn.BatchNorm2d(features_dim[i + 1]))
            self.encod.append(nn.LeakyReLU())
        self.encod.append(nn.Flatten())
        self.h_dim = get_output_shape(self.encod, (3, input_dim, i_w, i_h))[1]
        self.mu_layer = nn.Linear(self.h_dim, zd_dim)
        self.log_sigma2_layer = nn.Linear(self.h_dim, zd_dim)

    def forward(self, x):
        """
        :param x: input data, assumed to have 3 channels, but only the first one is passed through the network.
        """
        z = self.encod(x)
        mu = self.mu_layer(z)
        log_sigma2 = self.log_sigma2_layer(z)
        return mu, log_sigma2


class ConvolutionalDecoder(nn.Module):
    def __init__(self, zd_dim, h_dim, input_dim=3, features_dim=[32, 64, 128]):  # , 256, 512, 1024]):
        """
        VAE Decoder
        :param zd_dim: dimension of the latent space
        :param input_dim: dimension of the oritinal input / output reconstruction, e.g., (28, 28) for MNIST
        :param features_dim: list of dimensions of the hidden layers
        """
        super(ConvolutionalDecoder, self).__init__()
        # e = ConvolutionalEncoder(zd_dim, input_dim=3, features_dim=features_dim)
        self.linear = nn.Linear(zd_dim, h_dim)
        self.sigmoid_layer = nn.Sigmoid()
        self.unflat = UnFlatten(features_dim[-1])
        self.decod = nn.Sequential()
        features_dim = [input_dim] + features_dim
        features_dim.reverse()
        k = [3, 4, 4]
        for i in range(len(features_dim) - 2):
            self.decod.append(
                nn.ConvTranspose2d(features_dim[i], features_dim[i + 1], kernel_size=k[i], stride=2, padding=1))
            self.decod.append(nn.BatchNorm2d(features_dim[i + 1]))
            self.decod.append(nn.LeakyReLU())
        self.decod.append(nn.ConvTranspose2d(features_dim[-2], input_dim * 2, kernel_size=k[-1], stride=2, padding=1))

    def forward(self, z):
        """
        :param z: latent space representation
        :return x_pro: reconstructed data, which is assumed to have 3 channels, but the channels are assumed to be equal to each other.
        """

        z = self.linear(z)
        z = self.unflat(z)
        x_decoded = self.decod(z)
        x_pro = self.sigmoid_layer(x_decoded[:, 0:3, :, :])
        log_sigma = x_decoded[:, 3:, :, :]

        return x_pro, log_sigma