import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import os
import itertools
# import toml
from sklearn.mixture import GaussianMixture
import numpy as np
# from sklearn.mixture import GaussianMixture
# import tqdm
from domainlab.utils.utils_class import store_args
from domainlab.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv import \
    DecoderConcatLatentFCReshapeConvGatedConv
from domainlab.compos.vae.compos.encoder import LSEncoderDense
from domainlab.models.a_model_classif import AModelClassif
from domainlab.utils.utils_classif import logit2preds_vpic, get_label_na
# from torch.optim import Adam
# from sklearn.metrics import accuracy_score
# from torch.optim.lr_scheduler import StepLR
# from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE
import torch.nn as nn
from domainlab.utils.utils_classif import logit2preds_vpic, get_label_na
from domid.compos.nn_net import Net_MNIST
import torch

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

# FIXME another builder: define blocks
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
    # print('fake img', model(torch.rand(*(image_dim))).data.shape)
    return model(torch.rand(*(image_dim))).data.shape


def cnn_encoding_block(in_c, out_c, kernel_size=(4, 4), stride=2, padding=1):
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU()  # negative slope
    ]
    return layers


def cnn_encoding_block(in_c, out_c, kernel_size=(4, 4), stride=2, padding=1):
    nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
    nn.BatchNorm2d(out_c)
    nn.LeakyReLU()  # negative slope


def cnn_decoding_block(in_c, out_c, kernel_size=(3, 3), stride=2, padding=1):
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
        N = int(np.sqrt(input.shape[1] / filter_size))
        return input.view(input.size(0), filter_size, N, N)  # FIXME (3,3)


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
        # breakpoint()
        self.encod = nn.Sequential()

        features_dim = [input_dim] + features_dim
        k = [3, 3, 3]  # , 3, 3, 3]
        for i in range(len(features_dim) - 1):
            self.encod.append(nn.Conv2d(features_dim[i], features_dim[i + 1], kernel_size=k[i], stride=2, padding=1))
            self.encod.append(nn.BatchNorm2d(features_dim[i + 1]))
            self.encod.append(nn.LeakyReLU())
        self.encod.append(nn.Flatten())
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        # breakpoint()
        # nn.init.xavier_uniform_(self.encod[0].weight.data, gain=np.sqrt(2))
        # nn.init.orthogonal_(self.encod[0].bias.data, gain=np.sqrt(2))
        # self.encod = nn.Sequential(
        #     *cnn_encoding_block(input_dim, features_dim[0]),
        #     *cnn_encoding_block(features_dim[0], features_dim[1]),
        #     *cnn_encoding_block(features_dim[1], features_dim[2]),
        #     nn.Flatten()  # [batch size, filter,3, 3, 3]
        #
        # )

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
        # h_filter = get_output_shape(UnFlatten(), (batch_size, h_dim))#batch size!!!!!!!!!
        # print(h_filter)
        self.unflat = UnFlatten(features_dim[-1])
        self.decod = nn.Sequential()
        features_dim = [input_dim] + features_dim
        features_dim.reverse()
        # breakpoint()
        k = [3, 4, 4]
        # k = [4, 3, 3, 3, 4, 4]
        for i in range(len(features_dim) - 2):
            self.decod.append(
                nn.ConvTranspose2d(features_dim[i], features_dim[i + 1], kernel_size=k[i], stride=2, padding=1))
            self.decod.append(nn.BatchNorm2d(features_dim[i + 1]))
            self.decod.append(nn.LeakyReLU())
        #

        self.decod.append(nn.ConvTranspose2d(features_dim[-2], input_dim * 2, kernel_size=k[-1], stride=2, padding=1))
        # self.decod.append(nn.Sigmoid())
        # nn.init.xavier_uniform_(self.decod[0].weight.data, gain=np.sqrt(2))
        #
        # self.decod = nn.Sequential(
        #
        #     *cnn_decoding_block(features_dim[-1], features_dim[1], kernel_size=(4, 4)),
        #     *cnn_decoding_block(features_dim[1], features_dim[0], kernel_size=(5, 5)),
        #     *cnn_decoding_block(features_dim[0], input_dim, kernel_size=(6, 6)),
        #     nn.Sigmoid()
        # )

        # sizee = get_output_shape(self.decod, (3, input_dim, 100, 100))[1]
        # self.mu_layer = nn.Linear(30000, 100)  # input_dim[0], input_dim[1])#shape of the [ 3, 28, 28] FIXME
        # self.log_sigma2_layer = nn.Linear(30000, 100)  # input_dim[0], input_dim[1])

    def forward(self, z):
        """
        :param z: latent space representation
        :return x_pro: reconstructed data, which is assumed to have 3 channels, but the channels are assumed to be equal to each other.
        """

        z = self.linear(z)
        z = self.unflat(z)
        x_decoded = self.decod(z)
        #breakpoint()
        #print('here')
        # flatten x_pro and    hange the in of linear layer
        #x_flatten = x_decoded.view(x_decoded.shape[0], x_decoded.shape[1] * x_decoded.shape[2] * x_decoded.shape[3])
        #print('here')
        x_pro = self.sigmoid_layer(x_decoded[:, 0:3, :, :])
        #print('here')
        #log_sigma = self.log_sigma2_layer(x_flatten)
        #print('here')
        log_sigma = x_decoded[:, 3:, :, :]


        #x_pro = torch.reshape(mu, (x_decoded.shape[0], x_decoded.shape[1], x_decoded.shape[2], x_decoded.shape[3]))
        # mean value anf mean for each pixel
        # treat mu as reconstruction
        # mu (bs x 30000)
        # x_pro reshaped mu
        return x_pro, log_sigma


class ModelVaDECNN(nn.Module):
    def __init__(self, zd_dim, d_dim, device, L, i_c, i_h, i_w):
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

        self.L = L
        self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, input_dim=i_c, i_w=i_w, i_h=i_h).to(device)
        self.decoder = ConvolutionalDecoder(zd_dim=zd_dim, h_dim=self.encoder.h_dim, input_dim=i_c).to(device)

        self.log_pi = nn.Parameter(torch.FloatTensor(self.d_dim, ).fill_(1.0 / self.d_dim).log(),
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

        # pi = self.log_pi.exp()
        pi = F.softmax(self.log_pi, dim=0)
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
        x_pro, *_ = self.decoder(results[2])
        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)
        return preds, z_mu, z, log_sigma2_c, probs, x_pro

    def cal_loss(self, x, warmup_beta):
        """Function that is called in trainer_vade to calculate loss
        :param x: tensor with input data
        :return: ELBO loss
        """
        return self.ELBO_Loss(x, warmup_beta)

    def pretrain_loss(self, x):
        Loss = nn.HuberLoss()
        #Loss = nn.MSELoss()

        # provider = LossProvider()
        # loss_function = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        x_pro, *_ = self.decoder(z)
        # print('out', x_pro[0, 0, 0:5, 0:3])
        # print('in', x[0, 0, 0:5, 0:3])
        loss = Loss(x, x_pro)

        return loss

    def ELBO_Loss(self, x, warmup_beta):  # warmup_beta - 0-1
        """ELBO loss function

        Using SGVB estimator and the reparametrization trick calculates ELBO loss.
        Calculates loss between encoded input and input using ELBO equation (12) in the papaer.

        :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
        :param int L: Number of Monte Carlo samples in the SOVB
        """
        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self._inference(x)

        eps = 1e-10
        L_rec = 0.0

        for l in range(self.L):
            #print(l)

            # print('mean mu', torch.mean(mu), 'max mu', torch.max(mu))
            # print('mean log sigma', torch.mean(log_sigma), 'max log sigma', torch.max(log_sigma))
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu  # shape [batch_size, self.zd_dim]
            x_pro, log_sigma = self.decoder(z)
            try:

                #breakpoint()

                #Loss Version 1:
                #L_rec += torch.mean(torch.sum(torch.sum(torch.sum(-log_sigma, 2),2),1),0)-0.5*F.mse_loss(x, x_pro, reduction = "sum")/z.shape[0]

                # Loss Version 2:
                # L_rec += torch.sum(-log_sigma)/z.shape[0] - torch.sum(0.5 * (x - mu) ** 2 / torch.exp(log_sigma) ** 2)/z.shape[0]

                #Loss Version 3:
                #L_rec += 0.5*F.mse_loss(x, x_pro, reduction = "sum")/z.shape[0]

                # Loss Version 4:
                #print('log sigma sum', torch.mean(torch.sum(torch.sum(torch.sum(log_sigma, 2),2),1),0))
                #print('sigma squared', torch.mean(torch.sum(torch.sum(torch.sum(log_sigma ** 2, 2), 2), 1), 0))
                L_rec += torch.mean(torch.sum(torch.sum(torch.sum(0.5 * log_sigma**2, 2),2),1),0)\
                         +torch.mean(torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2 / torch.exp(log_sigma) ** 2, 2), 2), 1), 0)
                #print('built in', 0.5*F.mse_loss(x, x_pro, reduction = "sum"))
                #L_rec += torch.mean(torch.sum(torch.sum(torch.sum(0.5 * (x - x_pro) ** 2 / torch.exp(log_sigma) ** 2, 2), 2), 1), 0)
                # L_rec = L_rec*warmup_beta

                #Loss Version 5: added 1/constant_sigma_estimate to Version 1
                # sigma_estimate = 1/np.log(0.2) #0.2 - std from all the images
                #sigma_estimate = 1
                #L_rec += -torch.mean(torch.sum(torch.sum(torch.sum(log_sigma, 2), 2), 1), 0) - 0.5 * 1/sigma_estimate*F.mse_loss(x, x_pro,reduction="sum") /z.shape[0]

                #Loss Version 6: added 1/constant_sigma_estimate to Version 3
                # sigma_estimate = 1 / np.log(0.2)**2
                #sigma_estimate = (z.shape[0]/torch.sum(log_sigma))
                # sigma_estimate = 1
                # #warmup_beta = 1
                # L_rec += 0.5 *sigma_estimate*F.mse_loss(x, x_pro, reduction="sum") / z.shape[0]


                #Loss Version 7:
                # sigma = torch.exp(0.5 * log_sigma)
                # L_rec += (-0.5 * torch.log(2 * np.pi * torch.sum(sigma ** 2)) - (1 / (2 * torch.sum(sigma ** 2))) * torch.sum((x - mu) ** 2))/z.shape[0]





            except:
                print('loss is nan')
                breakpoint()

        if L_rec<0 or torch.isnan(L_rec):
            breakpoint()
        print('Reconstuction loss', L_rec)
        print('_'*10)
        # TODO: this is the reconstruction loss for a binary-valued x (such as MNIST digits); need to implement another version for a real-valued x.

        L_rec /= self.L

        Loss = L_rec

        # doesn't take the mean over the channels; i.e., the recon loss is taken as an average over (batch size * L * width * height)
        # --> this is the -"first line" of eq (12) in the paper with additional averaging over the batch.

        # warmUp beta is multiplied here and incresing to 1
        #print('Checkpoint 1 (reconstruction)', Loss)
        Loss += 0.5 * warmup_beta * torch.mean(
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
        #print('Checkpoint 2', Loss)
        Loss -= warmup_beta * torch.mean(torch.sum(probs * torch.log(pi.unsqueeze(0) / (probs + eps)),
                                     1))  # FIXME: (+eps) is a hack to avoid NaN. Is there a better way?
        # dimensions: [batch_size, d_dim] * log([1, d_dim] / [batch_size, d_dim]), where the sum is over d_dim dimensions --> [batch_size] --> mean over the batch --> a scalar
        # print('chepoint 3', Loss)
        Loss -= 0.5 * warmup_beta * torch.mean(torch.sum(1.0 + z_sigma2_log, 1))
        #print('Checkpoint 4', Loss)
        #print('_________________________________')

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
    # print(torch.__version__)
    i_w = 28
    i_h = 28
    model_cnn = ModelVaDECNN(y_dim=y_dim, zd_dim=zd_dim, device=torch.device("cpu"), i_w=i_w, i_h=i_h)
    device = torch.device("cpu")

    batch_size = 5
    x = torch.rand(batch_size, 3, i_w, i_h)
    a = np.zeros((batch_size, y_dim))
    a = np.double(a)
    a[0, 1] = 1.0
    a[1, 8] = 1.0
    a
    y = torch.tensor(a, dtype=torch.float)
    print('x, y', x.shape, y.shape)
    model_cnn.encoder(x)
    loss = model_cnn.cal_loss(x)
    print(loss)
