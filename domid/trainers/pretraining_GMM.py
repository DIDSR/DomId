import numpy as np
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture


class Pretraining:
    def __init__(self, model, device, loader_tr, loader_val, i_h, i_w, args):
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
        self.args = args
        self.is_inject_domain = False
        # if self.args.dim_inject_y > 0:
        #     self.is_inject_domain = True

    def pretrain_loss(self, tensor_x, inject_tensor):
        """
        :param tensor_x: the input image
        :return: the loss
        """
        # import pdb; pdb.set_trace()
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
            for tensor_x, vec_y, vec_d, *other_vars in self.loader_tr:
                if len(other_vars) > 0:
                    inject_tensor, image_id = other_vars
                    if len(inject_tensor) > 0:
                        inject_tensor = inject_tensor.to(self.device)

                tensor_x, vec_y, vec_d = (
                    tensor_x.to(self.device),
                    vec_y.to(self.device),
                    vec_d.to(self.device),
                )

                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x, inject_tensor)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                Z[counter : counter + z.shape[0], :] = z
                counter += z.shape[0]

        try:
            gmm = GaussianMixture(
                n_components=self.model.d_dim,
                covariance_type="diag",
                reg_covar=10**-3,
            )  # , reg_covar=10)
            out_fit_pred = gmm.fit_predict(Z)
            # print(out_fit_pred)
        except Exception as ex:
            raise RuntimeError("Gaussian mixture model failed:" + str(ex))
        # visitor/intruder to deep learning model to change model parameters
        self.model.log_pi.data = torch.log(torch.from_numpy(gmm.weights_)).to(self.device).float()
        # name convention: mu_c is the mean for the Gaussian mixture cluster,
        # but mu alone means mean for decoded pixel
        self.model.mu_c.data = torch.from_numpy(gmm.means_).to(self.device).float()
        self.model.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_)).to(self.device).float()
