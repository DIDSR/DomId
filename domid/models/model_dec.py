import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.utils.utils_classif import logit2preds_vpic

from domid.compos.DEC_clustering_layer import DECClusteringLayer
from domid.models.a_model_cluster import AModelCluster

def mk_dec(parent_class=AModelCluster):

    class ModelDEC(parent_class):
        def __init__(self, zd_dim, d_dim, L, device, i_c, i_h, i_w, args):
            """
            DEC model (Xie et al. 2015 "Unsupervised Deep Embedding for Clustering Analysis") with
            fully connected encoder and decoder.

            :param zd_dim: dimension of the latent space
            :param d_dim: number of clusters for the clustering task
            :param device: device to use, e.g., "cuda" or "cpu"
            :param i_c: number of channels of the input image
            :param i_h: height of the input image
            :param i_w: width of the input image
            :param args: command line arguments
            """
            super(ModelDEC, self).__init__()
            self.n_clusters = d_dim
            self.d_dim = d_dim
            self.zd_dim = zd_dim
            self.device = device

            self.alpha = 1  # FIXME
            self.hidden = zd_dim
            self.cluster_centers = None
            self.dim_inject_y = 0
            self.warmup_beta = 0
            self.args = args

            if self.args.feat_extract == "vae":
                from domid.compos.cnn_VAE import ConvolutionalDecoder, ConvolutionalEncoder
            elif self.args.feat_extract == "ae":
                from domid.compos.cnn_AE import ConvolutionalDecoder, ConvolutionalEncoder

            self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, num_channels=i_c, i_w=i_w, i_h=i_h).to(device)
            self.decoder = ConvolutionalDecoder(
                prior=args.prior,
                zd_dim=zd_dim,  # 50
                domain_dim=self.dim_inject_y,  #
                # domain_dim=self.dim_inject_y,
                h_dim=self.encoder.h_dim,
                num_channels=i_c,
            ).to(device)

            self.autoencoder = self.encoder
            self.clusteringlayer = DECClusteringLayer(
                self.n_clusters, self.hidden, None, self.alpha, self.device
            )  # learnable parameter - cluster center
            self.mu_c = self.clusteringlayer.cluster_centers
            self.log_pi = nn.Parameter(
                torch.FloatTensor(
                    self.d_dim,
                )
                .fill_(1.0 / self.d_dim)
                .log(),
                requires_grad=True,
            )
            if self.args.pre_tr_weight_path:
                self.encoder.load_state_dict(
                    torch.load(self.args.pre_tr_weight_path + "encoder.pt", map_location=self.device)
                )
                self.decoder.load_state_dict(
                    torch.load(self.args.pre_tr_weight_path + "decoder.pt", map_location=self.device)
                )
                print("AE is using pretrained weights. No need for pretraining epochs. ")
            self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.d_dim, self.zd_dim).fill_(0), requires_grad=True)
            self.random_ind = []

        def target_distribution(self, q_):
            """
            Corresponds to equation 3 from the paper.
            Calculates the target distribution for the Kullback-Leibler divergence loss.

            :param q_: A tensor of the predicted cluster probabilities.
            :return tensor: The calculated target distribution
            """
            weight = (q_**2) / torch.sum(q_, 0)
            return (weight.t() / torch.sum(weight, 1)).t()

        def _inference(self, x):
            """
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

            z_mu = self.encoder.get_z(x)
            z_sigma2_log = self.encoder.get_log_sigma2(x)

            probs_c = self.clusteringlayer(z_mu)  # in dec it is
            preds_c, logits, *_ = logit2preds_vpic(probs_c)  # preds c is oen hot encoded
            mu_c = self.mu_c
            # print(mu_c[0, :5])
            log_sigma2_c = self.log_sigma2_c
            pi = self.log_pi

            return preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits


        def infer_d_v_2(self, x, inject_domain):

            results = self._inference(x)
            if len(inject_domain) > 0:
                zy = torch.cat((results[2], inject_domain), 1)
            else:
                zy = results[2]

            x_pro, *_ = self.decoder(zy)
            preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = (r.cpu().detach() for r in results)
            return preds, z_mu, z, log_sigma2_c, probs, x_pro



        def _cal_kl_loss(self, x, inject_tensor, warmup_beta=0.1):
            """
            Calculates the KL-divergence loss between the predicted probabilities and the target distribution.

            :param x: input tensor/image
            :param inject_tensor: tensor to inject (not used in DEC, only used in CDVaDE
            :param warmup_beta: warm-up beta value
            :return tensor loss (float): calculated KL-divergence loss value
            """

            preds, probs, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self._inference(x)

            target = self.target_distribution(probs).detach()
            loss_function = nn.KLDivLoss(reduction="batchmean")

            loss = loss_function(probs.log(), target)
            if self.warmup_beta != warmup_beta:
                print(logits[0, :], target[0, :])
                print(loss)
                self.warmup_beta = warmup_beta
            return loss

    return ModelDEC