import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_classif import logit2preds_vpic
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter

from domid.compos.cnn_AE import ConvolutionalDecoder, ConvolutionalEncoder
from domid.compos.linear_AE import LinearDecoderAE, LinearEncoderAE
from domid.models.a_model_cluster import AModelCluster
from domid.models.a_model_cluster import AModelCluster
def mk_ae(parent_class=AModelCluster):

    class ModelAE(parent_class):
        def __init__(self, zd_dim, d_dim, device, L, i_c, i_h, i_w, args):

            super(ModelAE, self).__init__()
            self.zd_dim = zd_dim
            self.d_dim = d_dim
            self.device = device
            self.L = L
            self.args = args
            self.loss_epoch = 0
            self.batch_zero = True
            self.dim_inject_y = 0

            if self.args.dim_inject_y:
                self.dim_inject_y = self.args.dim_inject_y

            n_z = zd_dim
            n_input = i_c * i_h * i_w
            n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, = (
                500,
                500,
                2000,
                2000,
                500,
                500,
            )

            self.cluster_layer = nn.Parameter(torch.Tensor(self.d_dim, self.zd_dim))
            torch.nn.init.xavier_normal_(self.cluster_layer.data)

            if self.args.model == "linear":
                self.encoder = LinearEncoderAE(n_enc_1, n_enc_2, n_enc_3, n_input, n_z)
                self.decoder = LinearDecoderAE(n_dec_1, n_dec_2, n_dec_3, n_input, n_z)

            else:
                self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, num_channels=i_c, i_w=i_w, i_h=i_h).to(device)
                self.decoder = ConvolutionalDecoder(
                    prior=args.prior,
                    zd_dim=zd_dim,  # 50
                    domain_dim=self.dim_inject_y,  #
                    # domain_dim=self.dim_inject_y,
                    h_dim=self.encoder.h_dim,
                    num_channels=i_c,
                ).to(device)
            if self.args.pre_tr_weight_path:
                self.encoder.load_state_dict(
                    torch.load(self.args.pre_tr_weight_path + "encoder.pt", map_location=self.device)
                )
                self.decoder.load_state_dict(
                    torch.load(self.args.pre_tr_weight_path + "decoder.pt", map_location=self.device)
                )
                print("Pre-trained weights loaded")

            self.counter = 0
            ex = str(datetime.now())  # .strftime("%H:%M")
            self.local_tb = SummaryWriter(log_dir=os.path.join("local_tb", ex))
            self.random_ind = []

        def distance_between_clusters(self, cluster_layer):

            pairwise_dist = torch.zeros(cluster_layer.shape[0], cluster_layer.shape[0])
            for i in range(0, cluster_layer.shape[0]):
                for j in range(0, cluster_layer.shape[0]):
                    pairwise_dist[i, j] = torch.cdist(
                        cluster_layer[i, :].unsqueeze(0).unsqueeze(0), cluster_layer[j, :].unsqueeze(0).unsqueeze(0)
                    )
            return pairwise_dist

        def _inference(self, x, inject_tensor=None):

            if self.args.model == "linear":
                x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            enc_h1, enc_h2, enc_h3, z = self.encoder(x)
            # _, _, z, *_ = self._inference(x)

            kmeans = KMeans(n_clusters=self.args.d_dim, n_init=20)

            kmeans.fit_predict(z.detach().cpu().numpy())
            # x_bar, *_ = self.decoder(z)
            z_mu = torch.mean(z, dim=0)
            z_sigma2_log = torch.std(z, dim=0)
            pi = torch.Tensor([0])
            predictions = kmeans.labels_
            preds_c = mk_fun_label2onehot(self.d_dim)(predictions)
            logits = torch.Tensor(kmeans.fit_transform(z.detach().cpu().numpy())).to(self.device)
            _, probs_c, *_ = logit2preds_vpic(logits)
            cluster_layer = torch.tensor(kmeans.cluster_centers_)

            return preds_c, probs_c, z, z_mu, z_sigma2_log, z_mu, z_sigma2_log, pi, logits



        def infer_d_v_2(self, x, inject_domain):
            """
            Used for tensorboard visualizations only.
            """
            results = self._inference(x)
            if len(inject_domain) > 0:
                zy = torch.cat((results[2], inject_domain), 1)
            else:
                zy = results[2]
            x_pro = self.decoder(zy)

            preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)

            return preds, z_mu, z, log_sigma2_c, probs, x_pro

        def _cal_loss(self, x, inject_domain, warmup_beta=None):
            loss = self._cal_pretrain_loss(x, inject_domain)
            return loss



    return ModelAE


def test_fun(d_dim, zd_dim, device):
    device = torch.device("cpu")
    model = ModelAE(d_dim=d_dim, zd_dim=zd_dim, device=device)
    x = torch.rand(2, 3, 28, 28)
    import numpy as np

    a = np.zeros((2, 10))
    a = np.double(a)
    a[0, 1] = 1.0
    a[1, 8] = 1.0
    a
    y = torch.tensor(a, dtype=torch.float)
    model(x, y)
    model.cal_loss(x)
