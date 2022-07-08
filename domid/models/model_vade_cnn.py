import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import os
import itertools
# import toml
from sklearn.mixture import GaussianMixture
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
        print(self.filter3)
    def forward(self, input):
        filter_size = self.filter3 #FIXME same as filter 3
        n = int(np.sqrt(input.shape[1]/filter_size))
        return input.view(input.size(0), filter_size, 3, 3)#FIXME (3,3)

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
        self.encod = nn.Sequential(
            *cnn_encoding_block(input_dim, features_dim[0]),
            *cnn_encoding_block(features_dim[0], features_dim[1]),
            *cnn_encoding_block(features_dim[1], features_dim[2]),
            nn.Flatten()  # [batch size, filter,3, 3, 3]

        )
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
    def __init__(self, zd_dim, input_dim=3, features_dim=[32, 64, 128], h_dim =1152): #FIXME
        """
        VAE Decoder
        :param zd_dim: dimension of the latent space
        :param input_dim: dimension of the oritinal input / output reconstruction, e.g., (28, 28) for MNIST
        :param features_dim: list of dimensions of the hidden layers
        """
        super(ConvolutionalDecoder, self).__init__()
        self.linear = nn.Linear(zd_dim, h_dim)

        # h_filter = get_output_shape(UnFlatten(), (batch_size, h_dim))#batch size!!!!!!!!!
        # print(h_filter)
        self.unflat = UnFlatten(128) #FIXME 
        self.decod = nn.Sequential(

            *cnn_decoding_block(features_dim[-1], features_dim[1], kernel_size=(4, 4)),
            *cnn_decoding_block(features_dim[1], features_dim[0], kernel_size=(5, 5)),
            *cnn_decoding_block(features_dim[0], input_dim, kernel_size=(6, 6)),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        :param z: latent space representation
        :return x_pro: reconstructed data, which is assumed to have 3 channels, but the channels are assumed to be equal to each other.
        """
        z = self.linear(z)
        z = self.unflat(z)
        x_pro = self.decod(z)


        return x_pro


class ModelVaDECNN(nn.Module):
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

        super(ModelVaDECNN, self).__init__()
        self.zd_dim = zd_dim
        self.d_dim = d_dim
        self.device = device
        self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, input_dim=i_c).to(device)
        self.decoder = ConvolutionalDecoder(zd_dim=zd_dim, input_dim=i_c).to(device)

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
        breakpoint()
        if np.all(self.pi_.detach().cpu().numpy()==self.pi_[0].detach().cpu().numpy()):
            preds, probs_c, *_ = logit2preds_vpic(z)

            mu_c, log_sigma2_c, pi, logits = None

        else:
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
        preds, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self._inference(x)



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

    def pretrain_loss(self, x):
        Loss = nn.MSELoss()
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        x_pro = self.decoder(z)
        loss = Loss(x, x_pro)


        return loss

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

def test_fun(y_dim, zd_dim, device):
    #print(torch.__version__)
    i_w =28
    i_h =28
    model_cnn = ModelVaDECNN(y_dim=y_dim, zd_dim=zd_dim, device=torch.device("cpu"), i_w=i_w, i_h =i_h)
    device = torch.device("cpu")

    batch_size =5
    x = torch.rand(batch_size, 3, i_w, i_h)
    a = np.zeros((batch_size, y_dim))
    a = np.double(a)
    a[0, 1] = 1.0
    a[1, 8] = 1.0
    a
    y = torch.tensor(a, dtype=torch.float)
    print('x, y', x.shape, y.shape)
    model_cnn.encoder(x)
    loss = model_cnn.cal_loss(x, zd_dim)
    print(loss)
