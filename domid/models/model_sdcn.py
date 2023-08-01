import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.utils.utils_classif import logit2preds_vpic
from tensorboardX import SummaryWriter

from domid.compos.cnn_AE import ConvolutionalDecoder, ConvolutionalEncoder
from domid.compos.linear_VAE import LinearDecoder, LinearEncoder
from domid.models.a_model_cluster import AModelCluster
from domid.compos.GNN_layer import GNNLayer
from domid.compos.linear_AE import LinearEncoderAE, LinearDecoderAE
from domid.compos.GNN import GNN

import scipy.sparse as sp
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from datetime import datetime
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
   

        self.cluster_layer = nn.Parameter(torch.Tensor(self.d_dim, self.zd_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


        if self.args.model == "linear":
            n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, = 500, 500, 2000, 2000, 500, 500,
            self.encoder =  LinearEncoderAE(n_enc_1, n_enc_2, n_enc_3,n_input, n_z)
            self.decoder = LinearDecoderAE(n_dec_1, n_dec_2, n_dec_3, n_input, n_z)

            
        else:
            self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, num_channels=i_c, i_w=i_w, i_h=i_h).to(device)
            self.decoder = ConvolutionalDecoder(
                prior=args.prior,
                zd_dim=zd_dim, #50
                domain_dim=self.dim_inject_y, #
                #domain_dim=self.dim_inject_y,
                h_dim=self.encoder.h_dim,
                num_channels=i_c
            ).to(device)
            n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, = int((i_w/2)**2*32), int((i_w/4)**2*64), int((i_w/8)**2*128), int((i_w/8)**2*128), int((i_w/4)**2*64), int((i_w/2)**2*32)
            print(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3)

        self.encoder.load_state_dict(torch.load(self.args.pre_tr_weight_path + 'encoder.pt', map_location=self.device))
        self.decoder.load_state_dict(torch.load(self.args.pre_tr_weight_path + 'decoder.pt', map_location=self.device))


        self.gnn_model = GNN(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, n_clusters)

        self.v = 1.0
        self.counter= 0
        self.q_activation = torch.zeros((10, 100))
        # ex = str(datetime.now())
        # self.local_tb = SummaryWriter(log_dir=os.path.join('local_tb',ex ))
        # self.batch_zero = True

        self.kl_loss_running = 0
        self.re_loss_running = 0
        self.ce_loss_running = 0


    def distance_between_clusters(self, cluster_layer):
        
        pairwise_dist = torch.zeros(cluster_layer.shape[0], cluster_layer.shape[0])
        for i in range(0, cluster_layer.shape[0]):
            for j in range(0,cluster_layer.shape[0]):
                pairwise_dist[i,j] = torch.cdist(cluster_layer[i, :].unsqueeze(0).unsqueeze(0), cluster_layer[j, :].unsqueeze(0).unsqueeze(0))
        return pairwise_dist


    def _inference(self, x, inject_tensor=None):
        if self.args.model == "linear":
            x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        enc_h1, enc_h2, enc_h3, z = self.encoder(x)

        h = self.gnn_model(x, self.adj.to(self.device), enc_h1, enc_h2, enc_h3, z)
        probs_c = F.softmax(h, dim=1) # [batch_size, n_clusters] (batch_zise==number of samples) same as preds in the code
        # and p is calculated using preds and target distribution.

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))/ self.v
        q = q.pow((self.v+ 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()


        logits = q.type(torch.float32) #q in the paper and code
        
        
        

        # if self.batch_zero:
        #     self.local_tb.add_histogram('clustering_layer', self.cluster_layer.flatten(), self.counter)
        #     self.local_tb.add_histogram('q', q.flatten(), self.counter)
        #     self.local_tb.add_histogram('pred', probs_c.flatten(), self.counter)
        #     self.local_tb.add_histogram('h', h.flatten(), self.counter)
        #     self.local_tb.add_histogram('z', z.flatten(), self.counter)


        z_mu =torch.mean(z, dim=0) # is not used in SDCN (variance from the encoder in VaDE)
        z_sigma2_log = torch.std(z, dim=0) # is not used in SDCN (variance from the encoder in VaDE)
        pi = torch.Tensor([0]) # is not used in SDCN (variance from the encoder in VaDE)

        # preds_c = torch.argmax(logits, dim=1)
        # preds_c = F.one_hot(preds_c, num_classes=self.d_dim)

        preds_c, *_ = logit2preds_vpic(h) # probs_c is F.softmax(logit, dim=1)
#         if self.batch_zero:
#             d = self.distance_between_clusters(self.cluster_layer.detach().cpu())
#             self.batch_zero = False
          
#             plt.imshow(d)
#             plt.colorbar()
#             plt.title('Epoch '+str(self.counter))
#             plt.show()
#             plt.savefig('./local_tb/SDCN_epoch_'+str(self.counter)+'.png')
#             plt.close()

        return preds_c, probs_c, z, z_mu, z_sigma2_log, z_mu, z_sigma2_log, pi, logits



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


    def cal_loss(self, x, inject_domain, warmup_beta=None):

        preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits= self._inference(x)
        # logits is q in the paper
        # probs_c is pred in the code
        q = logits
        pred = probs_c
        if len(inject_domain) > 0:
            
            zy = torch.cat((z, inject_domain), 1)
        else:
            zy = z
        x_bar, *_ = self.decoder(zy)
        q = q.data
        

        # if self.counter==1:
        p = self.target_distribution(q)

        # self.local_tb.add_histogram('p', p, self.counter)
        
        if self.args.model == "linear":
            x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x, x_bar)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        self.kl_loss_running = kl_loss
        self.ce_loss_running = ce_loss
        self.re_loss_running = re_loss

        # if self.batch_zero:
        #     self.local_tb.add_scalar('kl_loss', kl_loss, self.counter)
        #     self.local_tb.add_scalar('ce_loss', ce_loss, self.counter)
        #     self.local_tb.add_scalar('re_loss', re_loss, self.counter)
        #     self.batch_zero = False

#         print('reconstruction loss', re_loss, 'kl_loss', kl_loss, 'ce_loss', ce_loss)
#         print('loss', loss)
#             self.counter+=1
        return loss.type(torch.double)
    def cal_loss_for_tensorboard(self):
        return self.kl_loss_running, self.ce_loss_running, self.re_loss_running


    def pretrain_loss(self, x, inject_domain):
        if self.args.model == "linear":
            x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        enc_h1, enc_h2, enc_h3, z = self.encoder(x)

        if len(inject_domain) > 0:
            zy = torch.cat((z, inject_domain), 1)
        else:
            zy = z
        x_pro, *_ = self.decoder(zy) #FIXME account for different number of outputs from decoder


        loss = F.mse_loss(x, x_pro)

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
