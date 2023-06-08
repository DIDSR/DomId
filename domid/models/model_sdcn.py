import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.utils.utils_classif import logit2preds_vpic
from tensorboardX import SummaryWriter

from domid.compos.cnn_VAE import ConvolutionalDecoder, ConvolutionalEncoder
from domid.compos.linear_VAE import LinearDecoder, LinearEncoder
from domid.models.a_model_cluster import AModelCluster
from domid.compos.GNN_layer import GNNLayer
from domid.compos.linear_AE import LinearEncoderAE, LinearDecoderAE
from domid.compos.GNN import GNN

import scipy.sparse as sp
import matplotlib.pyplot as plt

class ModelSDCN(AModelCluster):
    def __init__(self, zd_dim, d_dim, device, L, i_c, i_h, i_w, args):

        super(ModelSDCN, self).__init__()
        self.zd_dim = zd_dim
        self.d_dim = d_dim
        self.device = device
        self.L = L
        self.args = args
        self.loss_epoch = 0

        self.dim_inject_y = 0

        if self.args.dim_inject_y:
            self.dim_inject_y = self.args.dim_inject_y

        n_clusters = d_dim
        n_z = zd_dim
        n_input = i_c * i_h * i_w
        n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, = 500, 500, 2000, 2000, 500, 500,

        self.cluster_layer = nn.Parameter(torch.Tensor(self.d_dim, self.zd_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


        self.encoder =  LinearEncoderAE(n_enc_1, n_enc_2, n_enc_3,n_input, n_z)
        self.decoder = LinearDecoderAE(n_dec_1, n_dec_2, n_dec_3, n_input, n_z)
        self.gnn_model = GNN(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, n_clusters)

        self.v = 1.0
        # self.log_pi = nn.Parameter(
        #     torch.FloatTensor(
        #         self.d_dim,
        #     )
        #     .fill_(1.0 / self.d_dim)
        #     .log(),
        #     requires_grad=True,
        # )
        # self.mu_c = nn.Parameter(torch.FloatTensor(self.d_dim, self.zd_dim).fill_(0), requires_grad=True)
        # self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.d_dim, self.zd_dim).fill_(0), requires_grad=True)
        self.counter= 0





    def _inference(self, x):
        x = x.view(x.size(0), -1)
        tra1, tra2, tra3, z = self.encoder(x)

        h = self.gnn_model(x, self.adj, tra1, tra2, tra3, z)
        probs_c = F.softmax(h, dim=1) #[batch_size, n_clusters] (batch_zise==number of samples)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))/ self.v
        q = q.pow((self.v+ 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        q = (q.t() / torch.sum(q, 1)).t() # [batch_size, n_clusters]

        logits = q.type(torch.float32)



        z_mu =torch.mean(z, dim=0)#is not used in SDCN (variance from the encoder in VaDE)
        z_sigma2_log = torch.std(z, dim=0) #is not used in SDCN (variance from the encoder in VaDE)
        pi = torch.Tensor([0]) #is not used in SDCN (variance from the encoder in VaDE)

        preds_c, probs_c_, *_ = logit2preds_vpic(logits) #probs_c is F.softmax(logit, dim=1)

        return preds_c, probs_c, z, z_mu, z_sigma2_log, z_mu, z_sigma2_log, pi, logits

    def inference_pretraining(self, x, inject_tensor):
        # x = x.view(x.size(0), -1)
        # tra1, tra2, tra3, z = self.encoder(x)
        _, _, z, *_ = self._inference(x)


        x_bar, *_ = self.decoder(z)
        return x_bar, z

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
        if len(inject_domain) > 0:
            zy = torch.cat((results[2], inject_domain), 1)
        else:
            zy = results[2]

        # print(results[2].shape, inject_domain.shape, zy.shape)
        x_pro, *_ = self.decoder(zy)


        preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)
        return preds, z_mu, z, log_sigma2_c, probs, x_pro


    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


    def cal_loss(self, x, inject_domain, warmup_beta):
        x = x.view(x.size(0), -1)

        preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits= self._inference(x)

        q = logits
        pred = probs_c
        x_bar, *_ = self.decoder(z)
        q = q.data
        p = self.target_distribution(q)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x, x_bar)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
        print('reconstruction loss', re_loss, 'kl_loss', kl_loss, 'ce_loss', ce_loss)
        print('loss', loss)

        return loss.type(torch.double)

    def pretrain_loss(self, x, inject_domain):

        Loss = nn.MSELoss()
        # Loss = nn.MSELoss(reduction='sum')
        # Loss = nn.HuberLoss()

        x = x.view(x.size(0), -1)
        tra1, tra2, tra3, z = self.encoder(x)

        if len(inject_domain) > 0:
            zy = torch.cat((z, inject_domain), 1)
        else:
            zy = z
        x_pro, *_ = self.decoder(zy) #FIXME account for different number of outputs from decoder

        loss = Loss(x, x_pro)

        return loss




# def test_fun(d_dim, zd_dim, device):
#     device = torch.device("cpu")
#     model = ModelVaDE(d_dim=d_dim, zd_dim=zd_dim, device=device)
#     x = torch.rand(2, 3, 28, 28)
#     import numpy as np
#
#     a = np.zeros((2, 10))
#     a = np.double(a)
#     a[0, 1] = 1.0
#     a[1, 8] = 1.0
#     a
#     y = torch.tensor(a, dtype=torch.float)
#     model(x, y)
#     model.cal_loss(x)
