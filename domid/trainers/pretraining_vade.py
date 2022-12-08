import numpy as np
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture


class Pretraining():
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
        if self.args.dim_inject_y > 0:
            self.is_inject_domain = True

    def pretrain_loss(self, tensor_x, inject_tensor):
        """
        :param tensor_x: the input image
        :return: the loss
        """
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
                if len(other_vars)>0:
                    machine, img_locs, pred_domain = other_vars  #Because of the MNIST color is implemented rn

                tensor_x, vec_y, vec_d = (
                    tensor_x.to(self.device),
                    vec_y.to(self.device),
                    vec_d.to(self.device),
                )

                if not self.is_inject_domain:
                    inject_tensor = torch.tensor([], dtype=vec_y.dtype)
                else:
                    # pred_domain is from loader_tr, we decide here wether we inject
                    # both class label and domain label or only class label, or
                    # nothing
                    if len(pred_domain) > 1:
                        pred_domain = pred_domain.to(self.device)
                        if vec_y.shape[1] + pred_domain.shape[1] == self.args.dim_inject_y:
                            inject_tensor = torch.cat((vec_y, pred_domain), 1)
                        else:
                            raise ValueError("Dimension of vec_y and pred_domain does not match dim_inject_y")
                    else:
                        if vec_y.shape[1] == self.args.dim_inject_y:
                            inject_tensor = vec_y
                        else:
                            raise ValueError("Dimension of vec_y does not match dim_inject_y")
                    # convert to dtype of vec_y
                    inject_tensor = inject_tensor.to(vec_y.dtype)
                # only use the encoder to get latent representations, which is
                # later fed into GMM. (hint: infer_d_v_2 does use decoder but this is
                # not connected to computational graph)
                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x, inject_tensor)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                Z[counter:counter + z.shape[0], :] = z
                counter += z.shape[0]

        try:
            gmm = GaussianMixture(n_components=self.model.d_dim, covariance_type='diag', reg_covar = 10 ** -5) #, reg_covar=10)
            out_fit_pred = gmm.fit_predict(Z)
            # print(out_fit_pred)
        except Exception as ex:
            raise RuntimeError("Gaussian mixture model failed:"+str(ex))
        # visitor/intruder to deep learning model to change model parameters
        self.model.log_pi.data = torch.log(torch.from_numpy(gmm.weights_)).to(self.device).float()
        # name convention: mu_c is the mean for the Gaussian mixture cluster,
        # but mu alone means mean for decoded pixel
        self.model.mu_c.data = torch.from_numpy(gmm.means_).to(self.device).float()
        self.model.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_)).to(self.device).float()
