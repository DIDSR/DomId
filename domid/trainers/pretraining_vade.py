import numpy as np
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture


class Pretraining():
    def __init__(self, model, device, loader_tr, loader_val, i_h, i_w, args, is_inject_domain):
        """
        :param model: the model to train
        :param device: the device to use
        :param loader_tr: the training data loader
        :param i_h: image height
        :param i_w: image width
        """
        self.model = model
        self.device = device
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.i_h, self.i_w = i_h, i_w
        self.is_inject_domain = is_inject_domain
        self.args = args

    def pretrain_loss(self, tensor_x, inject_tensor):
        """
        :param tensor_x: the input image
        :return: the loss
        """


        #tensor_x = tensor_x.to(self.device)
        loss = self.model.pretrain_loss(tensor_x, inject_tensor)
        return loss


    def GMM_fit(self):
        """
        During pre-training we estimate pi, mu_c, and log_sigma2_c with a GMM at the end of each epoch.
        After pre-training these initial parameter values are used in the calculation of the ELBO loss,
        and are further updated with backpropagation like all other neural network weights.
        """
        num_img = len(self.loader_tr.dataset)
        Z = np.zeros((num_img, self.model.zd_dim))
        counter = 0
        with torch.no_grad():
            for tensor_x, vec_y, vec_d, machine, img_locs, pred_domain in self.loader_tr:
                tensor_x = tensor_x.to(self.device)


                if self.is_inject_domain:
                    if len(vec_y) + len(pred_domain) == self.args.dim_inject_y:
                        inject_tensor = torch.cat(vec_y, pred_domain)
                    elif len(vec_y) == self.args.dim_inject_y:
                        inject_tensor = vec_y
                else:
                    inject_tensor = []

                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x, inject_tensor)
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




