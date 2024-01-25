import torch
import torch.nn as nn
import torch.nn.functional as F
from domainlab.utils.utils_classif import logit2preds_vpic

from domid.compos.cnn_AE import ConvolutionalDecoder, ConvolutionalEncoder
from domid.compos.GNN import GNN
from domid.compos.linear_AE import LinearDecoderAE, LinearEncoderAE
from domid.models.a_model_cluster import AModelCluster

def mk_sdcn(parent_class=AModelCluster):
    class ModelSDCN(parent_class):
        """
        ModelSDCN is a class that implements the SDCN model.(Bo D et al. 2020)
        The model is composed of a convolutional encoder and decoder, a GNN and a clustering layer.
        """

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
                n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, = (
                    500,
                    500,
                    2000,
                    2000,
                    500,
                    500,
                )
                self.encoder = LinearEncoderAE(n_enc_1, n_enc_2, n_enc_3, n_input, n_z)
                self.decoder = LinearDecoderAE(n_dec_1, n_dec_2, n_dec_3, n_input, n_z)

            else:
                self.encoder = ConvolutionalEncoder(zd_dim=zd_dim, num_channels=i_c, i_w=i_w, i_h=i_h).to(device)
                self.decoder = ConvolutionalDecoder(
                    prior=args.prior,
                    zd_dim=zd_dim,  # 50
                    domain_dim=self.dim_inject_y,
                    h_dim=self.encoder.h_dim,
                    num_channels=i_c,
                ).to(device)
                n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, = (
                    int((i_w / 2) ** 2 * self.encoder.num_filters[0]),
                    int((i_w / 4) ** 2 * self.encoder.num_filters[1]),
                    int((i_w / 8) ** 2 * self.encoder.num_filters[2]),
                    int((i_w / 8) ** 2 * self.encoder.num_filters[2]),
                    int((i_w / 4) ** 2 * self.encoder.num_filters[1]),
                    int((i_w / 2) ** 2 * self.encoder.num_filters[0]),
                )
                print("Filter sizes for GNN", n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3)
            if self.args.pre_tr_weight_path:
                self.encoder.load_state_dict(
                    torch.load(self.args.pre_tr_weight_path + "encoder.pt", map_location=self.device)
                )
                self.decoder.load_state_dict(
                    torch.load(self.args.pre_tr_weight_path + "decoder.pt", map_location=self.device)
                )
                print("Pre-trained weights loaded")
            else:
                raise ValueError("Pre-trianed weight path is not provided")

            self.gnn_model = GNN(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, n_clusters, device)

            if torch.cuda.device_count() > 1:  # Check if multiple GPUs are available
                print("Using DataParallel with {} GPUs.".format(torch.cuda.device_count()))
                # self.gnn_model = torch.nn.DataParallel(self.gnn_model)  # Wrap the model for DataParallel
                self.encoder = torch.nn.DataParallel(self.encoder)
                self.decoder = torch.nn.DataParallel(self.decoder)

            else:
                if torch.cuda.device_count() == 1:
                    print("Using a single GPU.")
                else:
                    print("Using CPU(s).")

            self.v = 1.0
            self.counter = 0
            self.q_activation = torch.zeros((10, 100))
            self.kl_loss_running = 0
            self.re_loss_running = 0
            self.ce_loss_running = 0

            if "mnist" in self.args.task:
                self.graph_method = "heat"
            if "wsi" in self.args.task:
                self.graph_method = "patch_distance"
            if self.args.graph_method is not None:
                self.graph_method = args.graph_method

            if args.task == "wsi":
                self.random_ind = [torch.randint(0, self.args.bs, (int(self.args.bs / 3),)) for i in range(0, 66)]
            else:
                self.random_ind = []

        def _inference(self, x, inject_tensor=None):
            """
            :param x: [batch_size, n_channels, height, width]
            :return:
                - probs_c - [batch_size, n_clusters]
                - q - [batch_size, n_clusters]
                - z - [batch_size, n_z]
                - z_mu - [batch_size, n_z]
                - z_sigma2_log - [batch_size, n_z]
                - pi - [batch_size, n_clusters]
                - logits - [batch_size, n_clusters]
            """
            if self.args.model == "linear":
                x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            enc_h1, enc_h2, enc_h3, z = self.encoder(x)

            h = self.gnn_model(x, self.adj.to(self.device), enc_h1, enc_h2, enc_h3, z)
            probs_c = F.softmax(
                h, dim=1
            )  # [batch_size, n_clusters] (batch_zise==number of samples) same as preds in the code
            # and p is calculated using preds and target distribution.

            # Dual Self-supervised Module
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2)) / self.v
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

            logits = q.type(torch.float32)  # q in the paper and code

            preds_c, *_ = logit2preds_vpic(h)  # probs_c is F.softmax(logit, dim=1)

            return preds_c, probs_c, z, logits


        def infer_d_v_2(self, x):
            """
            Used for tensorboard visualizations only.
            """
            # import pdb; pdb.set_trace()
            results = self._inference(x)

            z = results[2]

            # print(results[2].shape, inject_domain.shape, zy.shape)
            x_pro = self.decoder(z)
            preds_c, probs_c, z, logits = (r.cpu().detach() for r in results)

            return preds_c, z, probs_c, x_pro

        def target_distribution(self, q):
            """
            Compute the target distribution p, where p_i = (sum_j q_j)^2 / sum_j^2 q_j.
            Corresponds to equation (12) from the paper.
            """
            weight = q**2 / q.sum(0)
            return (weight.t() / weight.sum(1)).t()

        def _cal_kl_loss(self, x, warmup_beta=None):
            """
            Compute the loss of the model.
            Concentrate two different objectives, i.e. clustering objective and classification objective, in one loss function.
            Corresponds to equation (15) in the paper.
            :param tensor x: Input tensor of a shape [batchsize, 3, horzintal dim, vertical dim].
            :param float warmup_beta: Warmup coefficient for the KL divergence term.
            :return tensor loss: Loss tensor.
            """

            preds_c, probs_c, z, logits = self._inference(x)
            # logits is q in the paper
            # probs_c is pred in the code
            q = logits
            pred = probs_c

            x_bar = self.decoder(z)
            q = q.data

            # if self.counter==1:
            p = self.target_distribution(q)

            # self.local_tb.add_histogram('p', p, self.counter)

            if self.args.model == "linear":
                x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))

            kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
            ce_loss = F.kl_div(pred.log(), p, reduction="batchmean")
            re_loss = F.mse_loss(x, x_bar)

            loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

            self.kl_loss_running = kl_loss
            self.ce_loss_running = ce_loss
            self.re_loss_running = re_loss

            return loss.type(torch.double)

        def cal_loss_for_tensorboard(self):
            return self.kl_loss_running, self.ce_loss_running, self.re_loss_running

        def hyper_init(self, functor_scheduler):
            """hyper_init.
            :param functor_scheduler:
            """
            return functor_scheduler(trainer=None)

        def hyper_update(self, epoch, fun_scheduler):
            """hyper_update.
            :param epoch:
            :param fun_scheduler: the hyperparameter scheduler object
            """
            dict_rst = fun_scheduler(
                epoch
            )  # the __call__ method of hy
            # perparameter scheduler
            self.alpha = dict_rst["alpha"]



    return ModelSDCN
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
