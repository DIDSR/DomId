import numpy as np
import torch
import torch.nn as nn

from domid.compos.VAE_blocks import linear_block
import torch.nn.functional as F


class LinearEncoderAE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3,
                 n_input, n_z):
        super(LinearEncoderAE, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)



        return enc_h1, enc_h2, enc_h3, z


class LinearDecoderAE(nn.Module):
    def __init__(self, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(LinearDecoderAE, self).__init__()

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, z):


        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar