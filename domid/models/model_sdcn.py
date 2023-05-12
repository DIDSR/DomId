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
from domid.compos.linear_AE import LinearAE
from domid.compos.GNN import GNN


class ModelSDCN(AModelCluster):


    # def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
    #              n_input, n_z, n_clusters, v=1):
    #     super(ModelSDCN, self).__init__()
    #
    #     # autoencoder for intra information
    #     self.ae_encoder = LinearEncoder()
    #     self.ae_decoder = LinearDecoder()
    #
    #     self.encoder = LinearEncoder(zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
    #     self.decoder = LinearDecoder(prior=args.prior, zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
    #
    #     self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
    #
    #     # GCN for inter information
    #     self.gnn_1 = GNNLayer(n_input, n_enc_1)
    #     self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
    #     self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
    #     self.gnn_4 = GNNLayer(n_enc_3, n_z)
    #     self.gnn_5 = GNNLayer(n_z, n_clusters)
    #
    #     # cluster layer
    #     self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
    #     torch.nn.init.xavier_normal_(self.cluster_layer.data)
    #
    #     # degree
    #     self.v = v
    #
    # def forward(self, x, adj):
    #     # DNN Module
    #     x_bar, tra1, tra2, tra3, z = self.ae(x)
    #
    #     sigma = 0.5
    #
    #     # GCN Module
    #     h = self.gnn_1(x, adj)
    #     h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
    #     h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
    #     h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
    #     h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
    #     predict = F.softmax(h, dim=1)
    #
    #     # Dual Self-supervised Module
    #     q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
    #     q = q.pow((self.v + 1.0) / 2.0)
    #     q = (q.t() / torch.sum(q, 1)).t()
    #
    #     return x_bar, q, predict, z
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

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
        self.linearAE =  LinearAE(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,n_input, n_z)
        self.gnn = GNN(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, n_clusters)
        self.v = 1


        # if self.args.model == "linear":
        #     self.encoder = LinearEncoder(zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
        #     self.decoder = LinearDecoder(prior=args.prior, zd_dim=zd_dim, input_dim=(i_c, i_h, i_w)).to(device)
        #     if self.dim_inject_y:
        #         warnings.warn("linear model decoder does not support label injection")
        # else:
        #     self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, num_channels=i_c, i_w=i_w, i_h=i_h).to(device)
        #     self.decoder = ConvolutionalDecoder(
        #         prior=args.prior,
        #         zd_dim=zd_dim,  # 50
        #         domain_dim=self.dim_inject_y,  #
        #         # domain_dim=self.dim_inject_y,
        #         h_dim=self.encoder.h_dim,
        #         num_channels=i_c
        #     ).to(device)
        # print(self.encoder)
        # print(self.decoder)
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
        #
        # self.loss_writter = SummaryWriter()


    def _inference(self, x):


        #z_mu, z_sigma2_log = self.encoder(x)
        x_bar, tra1, tra2, tra3, z = self.linearAE(x)
        adj = 'this should be a graph'
        h = self.gnn(x, adj, tra1, tra2, tra3, z)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        preds_c = predict
        probs_c = q
        z_mu =z
        z_sigma2_log = z
        mu_c = self.cluster_layer
        log_sigma2_c = self.v




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
    def kl_loss(self, q, p):
        """
        Compute the KL divergence between two distributions.
        """
        return F.kl_div(q, p, reduction='batchmean')

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


    def cal_loss(self, x, inject_domain, warmup_beta):
        x_bar, q, pred, _ = self._inference(x, adj)
        q = q.data
        p = self.target_distribution(q)

        kl_loss = self.kl_loss(q.log(), p)
        ce_loss = self.kl_loss(pred.log(), p)
        re_loss = F.mse_loss(x, x_bar)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        return

    def pretrain_loss(self, x, inject_domain):

        Loss = nn.MSELoss()
        # Loss = nn.MSELoss(reduction='sum')
        # Loss = nn.HuberLoss()
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        if len(inject_domain) > 0:
            zy = torch.cat((z, inject_domain), 1)
        else:
            zy = z
        x_pro, *_ = self.decoder(zy)

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
