import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.optim as optim
from domid.utils.perf_cluster import PerfCluster
class Pretraining():
    def __init__(self, model, device, loader_tr, i_h, i_w):
        #super().__init__(tensor_x, mse_n, epoch, model, device, optimizer)
        self.model = model
        self.device = device
        self.loader_tr = loader_tr
        self.i_h, self.i_w = i_h, i_w

    def pretrain_loss(self, tensor_x, mse_n,epoch):

        tensor_x = tensor_x.to(self.device)
        loss = self.model.pretrain_loss(tensor_x)
        return loss

    def prediction(self):
        num_img = len(self.loader_tr.dataset)
        Z = np.zeros((num_img, self.model.zd_dim))
        IMGS = np.zeros((num_img, 3, self.i_h, self.i_w))
        domain_labels = np.zeros((num_img, 1))
        machine_labels = []
        counter = 0
        with torch.no_grad():
            for tensor_x, vec_y, vec_d, *other_vars in self.loader_tr:
                if len(other_vars) > 0:
                    machine, image_loc = other_vars
                    for i in range(len(machine)):
                        machine_labels.append(machine[i])
                tensor_x = tensor_x.to(self.device)
                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                IMGS[counter:counter+z.shape[0], :, :, :] = tensor_x
                Z[counter:counter + z.shape[0], :] = z
                domain_labels[counter:counter + z.shape[0], 0] = torch.argmax(preds, 1)+1

                breakpoint()
                counter += z.shape[0]
                if counter==num_img-2:
                    breakpoint()
        return IMGS, Z, domain_labels, machine_labels

    def GMM_fit(self):
        #print('PIIII', self.model.log_sigma2_c.data)
        # During pre-training we estimate pi, mu_c, and log_sigma2_c with a GMM at the end of each epoch.
        # After pre-training these initial parameter values are used in the calculation of the ELBO loss,
        # and are further updated with backpropagation like all other neural network weights.
        num_img = len(self.loader_tr.dataset)
        Z = np.zeros((num_img, self.model.zd_dim))
        counter = 0
        with torch.no_grad():
            for tensor_x, vec_y, vec_d, *_ in self.loader_tr:
                tensor_x = tensor_x.to(self.device)
                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                Z[counter:counter + z.shape[0], :] = z
                counter += z.shape[0]

        try:
            gmm = GaussianMixture(n_components=self.model.d_dim, covariance_type='diag', reg_covar = 10 ** -5) #, reg_covar=10)

            pre = gmm.fit_predict(Z)
        except:
            breakpoint()
        self.model.log_pi.data = torch.log(torch.from_numpy(gmm.weights_)).to(self.device).float()
        self.model.mu_c.data = torch.from_numpy(gmm.means_).to(self.device).float()
        self.model.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_)).to(self.device).float()

        return gmm
    def epoch_val_acc(self):
        acc, conf = PerfCluster.cal_acc(self.model, self.loader_tr, self.device, max_batches=None) #FIXME change to validation loader
        return acc, conf



