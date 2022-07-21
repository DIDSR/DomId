import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.optim as optim
from domid.utils.perf_cluster import PerfCluster
class Pretraining():
    def __init__(self, model, device, loader_tr):
        #super().__init__(tensor_x, mse_n, epoch, model, device, optimizer)
        self.model = model
        self.device = device
        #self.optimizer = optimizer
        #self.epo_loss_tr = epo_loss_tr
        self.loader_tr = loader_tr



    def pretrain_loss(self, tensor_x, mse_n,epoch):
        #print('i was in pretrain epoch')
        tensor_x = tensor_x.to(self.device)

        #assert epoch<mse_n

        loss = self.model.pretrain_loss(tensor_x)

    #self.scheduler.step()

        return loss
            #print("LOOOOSSS", loss)

    def GMM_fit(self ):
        #print('PIIII', self.model.log_sigma2_c.data)
        # During pre-training we estimate pi, mu_c, and log_sigma2_c with a GMM at the end of each epoch.
        # After pre-training these initial parameter values are used in the calculation of the ELBO loss,
        # and are further updated with backpropagation like all other neural network weights.

        num_img = len(self.loader_tr.dataset)
        Z = np.zeros((num_img, self.model.zd_dim))
        counter = 0
        with torch.no_grad():
            for tensor_x, vec_y, vec_d in self.loader_tr:
                tensor_x = tensor_x.to(self.device)
                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x)
                z = z.detach().cpu().numpy() #[batch_size, zd_dim]
                #print('ZZZZZ', z)

                #print(Z.shape, counter, counter+z.shape[0])
                Z[counter:counter+z.shape[0], :] = z
                counter += z.shape[0]
        try:
        #breakpoint()
            gmm = GaussianMixture(n_components=self.model.d_dim, covariance_type='diag', reg_covar = 10 ** -5) #, reg_covar=10)

            pre = gmm.fit_predict(Z)
        except:
            breakpoint()
        self.model.log_pi.data = torch.log(torch.from_numpy(gmm.weights_)).to(self.device).float()
        self.model.mu_c.data = torch.from_numpy(gmm.means_).to(self.device).float()
        self.model.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_)).to(self.device).float()
        # gmm = gmm.fit(Z) #FIXME
        # print(gmm.weights_[1:3], '\n', gmm.means_[0, 1:3])

        return gmm
    def epoch_val_acc(self):
        acc, conf = PerfCluster.cal_acc(self.model, self.loader_tr, self.device, max_batches=None) #FIXME change to validation loader
        return acc, conf



