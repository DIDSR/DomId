import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import os
import itertools
# import toml
import numpy as np
#from sklearn.mixture import GaussianMixture
# import tqdm
from libdg.utils.utils_class import store_args
from libdg.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv import DecoderConcatLatentFCReshapeConvGatedConv
from libdg.compos.vae.compos.encoder import LSEncoderDense
from libdg.models.a_model_classif import AModelClassif
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na
#from torch.optim import Adam
#from sklearn.metrics import accuracy_score
#from torch.optim.lr_scheduler import StepLR
# from tensorboardX import SummaryWriter
#from sklearn.manifold import TSNE
import torch.nn as nn
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na
from domid.compos.nn_net import Net_MNIST
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
#FIXME another builder: define blocks
"""
input_img = Input(shape=(64, 64, 3)) 
x = Conv2D(32, (3, 3), padding='same', strides = (2,2))(input_img)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

x = Conv2D(16, (3, 3), padding='same', strides = (2,2))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x) 

x = Conv2D(16, (3, 3), padding='same', strides = (2,2))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

x = Conv2D(16, (3, 3), padding='same', strides = (2,2))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Flatten()(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)



# DECODER
direct_input = Input(shape=latentSize)
x = Dense((16*16*16))(direct_input) ##used to be Dense((16*16*16))(direct_input)
x = Reshape((16,16,16))(x)

x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)


x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)


decoded = Conv2D(3, (3, 3), padding='same', strides = (1,1), activation='sigmoid')(x)
"""

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return sum([w[ind[0], ind[1]] for counter in ind]) * 1.0 / Y_pred.size  # , w[0]


def block_encoding(in_c, out_c, kernel_size=(5,5), stride=1, padding=0):
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU() #negative slope
    ]
    return layers

def block_decoding(in_c, out_c, kernel_size=(5,5), stride=1, padding=0):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU()
    ]
    return layers


class Encoder(nn.Module):
    def __init__(self, z_dim, dim_input=3, filter1=3, filter2=3, filter3=3):
        super(Encoder, self).__init__()
        self.encod = nn.Sequential(
            *block_encoding(dim_input, filter1),
            *block_encoding(filter1, filter2),
            *block_encoding(filter2, filter3)
            #*block(filter3, filter3)
        )

        flat_size = 32768 #FIXME find a way to calculate this
        self.mu_l = nn.Linear(flat_size, z_dim)
        self.log_sigma2_l = nn.Linear(flat_size, z_dim)

    def forward(self, x):
        print('encder forward', x[0, 1, 1:3, 1:3])

        #x = torch.reshape(x, (x.shape[0], 3, 28 * 28))
        #x = x[:, 0, :]
        e = self.encod(x)  # output shape: [batch_size, z_dim]
        print(e.shape)
        e = torch.flatten(e, start_dim=2)
        e = torch.flatten(e, start_dim=1)

        print('e shape', e.shape)
        mu = self.mu_l(e)  # output shape: [batch_size, num_clusters]
        log_sigma2 = self.log_sigma2_l(e)  # same as mu shape

        return mu, log_sigma2


class Decoder(nn.Module):
    def __init__(self, z_dim, dim_input=3, filter1=3, filter2=3, filter3=3):
        super(Decoder, self).__init__()

        self.decod = nn.Sequential(
            #nn.Linear(dim_input, z_dim),
            *block_decoding(z_dim, filter3),
            *block_decoding(filter3, filter2),
            *block_decoding(filter2, filter1),#,
            #*block_decoding(filter1, dim_input)#,
            #nn.Linear(filter1, dim_input)
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Decoder input shape is [batch_size, 10]
        """
        print(z.shape)
        #z = torch.unsqueeze(z, 2)
        #z = torch.unsqueeze(z, 2)
        z = torch.zeros((2, 4, 16, 16))
        #breakpoint()
        print(z.shape)
        x_pro = self.decod(z)

        x_pro = x_pro [:, 0:3, :, :]
        #x_pro = torch.reshape(x_pro, (x_pro.shape[0], 1, 28, 28))
        #x_pro = torch.cat((x_pro, x_pro, x_pro), 1)
        #x_pro = x_p
        print('x_pro', x_pro[0, 1, 1:3, 1:3])
        print('decoder shape', x_pro.shape)
        return x_pro


class ModelVaDECNN(nn.Module):

    def __init__(self, y_dim, zd_dim, device):

        super(ModelVaDECNN, self).__init__()
        """
           :param tensor y_dim: number of original dataset clusters.
           :param tensor zd_dim: number of cluster domains
        """

        self.y_dim = y_dim  # nClusters
        self.zd_dim = zd_dim
        self.dim_y = y_dim
        self.device = device
        print(y_dim, device, zd_dim )
        input_dimention = 3 #28 * 28
        filter1 = 32
        filter2 = filter1*2
        filter3 = filter2*2

        self.infer_domain = Encoder(z_dim=zd_dim, dim_input=input_dimention, filter1=filter1, filter2=filter2, filter3=filter3).to(
            device)
        # self.encoder = self.infer_domain
        self.encoder = Encoder(z_dim=zd_dim, dim_input=input_dimention, filter1=filter1, filter2=filter2, filter3=filter3).to(device)
        self.decoder = Decoder(z_dim=zd_dim, dim_input=input_dimention, filter1=filter1, filter2=filter2, filter3=filter3).to(device)
        self.pi_ = nn.Parameter(torch.FloatTensor(zd_dim, ).fill_(1) / zd_dim, requires_grad=True)  # 1/ndomains
        self.mu_c = nn.Parameter(torch.FloatTensor(zd_dim, zd_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(zd_dim, zd_dim).fill_(0), requires_grad=True)

    def infer_d_v(self, x):
        """encoding function that is used in the perf_cluser.py
            yita corresponds to equation from the paper #10
           :param tensor x: Input tensor of a shape [batchsize, 3, horzintal feat, vertical feat].
           :return tensor prediction: One hot encoded tensor of the encoded domain clusters

           """
        det = 1e-10
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        nClusters = self.zd_dim  # FIXME
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(nClusters, z, mu_c,
                                                                               log_sigma2_c)) + det  # shape [batch_size, 10]
        yita = yita_c.cpu()
        prediction, *_ = logit2preds_vpic(yita)
        return prediction

    def ELBO_Loss(self, zd_dim, x, L=1):
        """Loss function

        Using SGVB estimator and the reparametrization trick calculates ELBO loss.
        Calculates loss between encoded input and input using ELBO equation (12) in the papaer.
        Calculates encoded vector inside.

        :param int zd_dim: Number of domain cluster cluster
        :param tensor x: Input tensor of a shape [batchsize, 3, horzintal feat, vertical feat].
        :param int L: Number of Monte Carlo samples in the SOVB

        """

        det = 1e-10
        L_rec = 0
        z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):  # not quite sure what the loop is for
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
            x_pro = self.decoder(z)
            L_rec += F.binary_cross_entropy(x_pro, x)  # why binary cross entropy?

        L_rec /= L
        Loss = L_rec * x.size(1)

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(zd_dim, z, mu_c, log_sigma2_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters
        Loss += 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                              torch.exp(
                                                                  z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(
                                                                      0)) +
                                                              (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(
                                                                  2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + z_sigma2_log, 1))
        # print(Loss)
        # print('loss out', Loss)
        return Loss

    def gaussian_pdfs_log(self, y_dim, x, mus, log_sigma2s):
        """
        helper function that used to perform reparametrization in ELBO los calculation
        """
        G = []

        for c in range(y_dim):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))

        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        """
        subhelper function just one gausian pdf log calculation, used as a basis for gaussia_pdfs_logs function

        :param x:
        :param mu: mean for the cluster distribution for reparametrization (?)
        :param log_sigma2: std of the cluster distribtion
        :return: cluster distribution of the shape of [batch_size, 1]
        """

        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))

    def infer_y_vpicn(self, tensor_x, device='cpu'):
        """
        This is just a simulation function of the classification function.
        :param tensor_x:
        :return:
        """

        pred = torch.zeros((100, 10))
        pred = pred.to(device)
        prob = 0
        ind = 0
        confidence = 0
        na_class = 0
        return pred, prob, ind, confidence, na_class

    def cal_loss(self, x, zd_dim):
        """ funciton that is called in th trainer_vade to calculate loss
        :param x: tensor with input data
        :param zd_dim: number of domain clusters
        :return: ELBO loss
        """

        loss = self.ELBO_Loss(zd_dim, x)
        loss = loss.cpu()
        return loss


def test_fun(y_dim, zd_dim, device):
    #import numpy as np
    #import torch
    print(torch.__version__)
    model_cnn = ModelVaDECNN(y_dim=y_dim, zd_dim=zd_dim, device=torch.device("cpu"))
    device = torch.device("cpu")
    #breakpoint()
    #print(model)


    x = torch.rand(2, 3, 28, 28)
    a = np.zeros((2, y_dim))
    a = np.double(a)
    a[0, 1] = 1.0
    a[1, 8] = 1.0
    a
    y = torch.tensor(a, dtype=torch.float)
    print('x, y', x.shape, y.shape)
    model_cnn.encoder(x)
    loss = model_cnn.cal_loss(x, zd_dim)
    print(loss)
