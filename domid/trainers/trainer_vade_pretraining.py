import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.optim as optim

from libdg.algos.trainers.a_trainer import TrainerClassif

from domid.utils.perf_cluster import PerfCluster


class TrainerVADE(TrainerClassif):
    def __init__(self, model, task, observer, device, writer, aconf=None):
        super().__init__(model, task, observer, device, aconf)
        self.optimizer = optim.Adam(self.model.parameters(), lr=aconf.lr)
        self.epo_loss_tr = None
        self.writer = writer

    def tr_epoch(self, epoch):
        self.model.train()
        self.epo_loss_tr = 0
        #breakpoint()
        mse_n = 10  # FIXME: maybe have a command line argument to specify mse_n and elbo_n
        elbo_n = 30
        #breakpoint()
        for i, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x = tensor_x.to(self.device)
            self.optimizer.zero_grad()
            if epoch < mse_n:
                loss = self.model.pretrain_loss(tensor_x)
                #print("LOOOOSSS", loss)
            else:
                loss = self.model.cal_loss(tensor_x)

            loss = loss.sum()
            #print("LOSS back", loss)
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.cpu().detach().item()

        # During pre-training we estimate pi, mu_c, and log_sigma2_c with a GMM at the end of each epoch.
        # After pre-training these initial parameter values are used in the calculation of the ELBO loss,
        # and are further updated with backpropagation like all other neural network weights.
        if epoch < mse_n:
            num_img = len(self.loader_tr.dataset)
            Z = np.zeros((num_img, self.model.zd_dim))
            counter = 0
            for tensor_x, vec_y, vec_d in self.loader_tr:
                tensor_x = tensor_x.to(self.device)
                preds, z_mu, z, _, _, x_pro = self.model.infer_d_v_2(tensor_x)
                z = z.detach().cpu().numpy() #[batch_size, zd_dim]
                Z[counter:counter+z.shape[0], :] = z
                counter += z.shape[0]

            #breakpoint()
            gmm = GaussianMixture(n_components=self.model.d_dim, covariance_type='diag', reg_covar=10 ** -4)
            # breakpoint()
            pre = gmm.fit_predict(Z)
            # gmm = gmm.fit(Z) #FIXME
            # print(gmm.weights_[1:3], '\n', gmm.means_[0, 1:3])

            self.model.pi_.data = torch.from_numpy(gmm.weights_).to(self.device).float()
            self.model.mu_c.data = torch.from_numpy(gmm.means_).to(self.device).float()
            self.model.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_)).to(self.device).float()


        #
        # from sklearn.metrics.pairwise import cosine_similarity
        # from scipy import sparse
        # from sklearn import metrics
        # #acc_tr_pool, conf_mat_tr = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)
        # #acc_val, conf_mat_val = PerfCluster.cal_acc(self.model, self.loader_te, self.device)
        # preds, z_mu, z, _, _, x_pro = self.model.infer_d_v_2(tensor_x)
        # z = z.cpu()
        # #self.writer.add_scalar("pooled train clustering acc: ", acc_tr_pool, epoch)
        # #self.writer.add_scalar("clustering validation acc: ", acc_val)
        # # import numpy as np
        # # #breakpoint()
        # # flat_a, flat_b = torch.flatten(tensor_x, 1), torch.flatten(x_pro,1)
        # # flat_a, flat_b = torch.flatten(flat_a, 1), torch.flatten(flat_b, 1)
        # # flat_a, flat_b = torch.flatten(flat_a, 1), torch.flatten(flat_b, 1)
        # #
        # # breakpoint()
        # #
        # # #a_sparse, b_sparse = sparse.csr_matrix(flat_a), sparse.csr_matrix(flat_b)
        # # flat_a = flat_a.detach().numpy()
        # # flat_b = flat_b.detach().numpy()
        # #
        # # sim_sparse = cosine_similarity(flat_a, flat_b, dense_output=False)
        # #
        # #
        # # from torchmetrics import Accuracy, F1Score
        # # acc = Accuracy()
        # # F1 = F1Score(num_classes=7)
        # # reconstruction_acc = np.mean(sim_sparse)
        # # clustering_acc = acc(torch.argmax(preds, 1), torch.argmax(vec_d, 1))
        #
        #
        #
        #
        # name = "Output of the decoder" + str(epoch)
        # imgs = torch.cat((tensor_x[0:8,:, :, :], x_pro[0:8,:, :, :],), 0)
        # self.writer.add_images(name, imgs, 0)
        #
        # if epoch<mse_n:
        #     self.writer.add_scalar('MSE loss', self.epo_loss_tr, epoch)
        # else:
        #     self.writer.add_scalar('ELBO loss', self.epo_loss_tr, epoch)
        #     #self.writer.add_scalar('Reconstraction Accuracy (cos similarity)', reconstruction_acc, epoch)
        #     #self.writer.add_scalar('Domain clustering acc', clustering_acc, epoch)
        # if epoch ==elbo_n:
        #
        #     class_labels = torch.argmax(vec_y, 1)
        #     self.writer.add_embedding(z, metadata= class_labels, label_img=x_pro) #FIXME set global trainer step

        flag_stop = self.observer.update(epoch)  # notify observer

        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)
        print("before training, model accuracy:", acc)

