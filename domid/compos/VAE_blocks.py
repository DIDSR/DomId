import torch.nn as nn
import torch
import numpy as np

def get_output_shape(model, image_dim):
    #print('fake img', model(torch.rand(*(image_dim))).data.shape)
    return model(torch.rand(*(image_dim))).data.shape

def cnn_encoding_block(in_c, out_c, kernel_size=(4,4), stride=2, padding=1):
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU() #negative slope
    ]
    return layers

def cnn_decoding_block(in_c, out_c, kernel_size=(3,3), stride=2, padding=1):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU()
    ]
    return layers

class UnFlatten(nn.Module):
    def __init__(self, filter3):
        super(UnFlatten, self).__init__()
        self.filter3 = filter3

    def forward(self, input):
        filter_size = self.filter3
        N = int(np.sqrt(input.shape[1]/filter_size))
        return input.view(input.size(0), filter_size, N, N)

def linear_block(in_c, out_c):
    layers = [nn.Linear(in_c, out_c), nn.ReLU(True)]
    return layers
