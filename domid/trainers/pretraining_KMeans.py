import numpy as np
import torch
from sklearn.cluster import KMeans


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

    def pretrain_loss(self, tensor_x, inject_tensor):
        """
        :param tensor_x: the input image
        :return: the loss
        """
        # import pdb; pdb.set_trace()
        loss = self.model.pretrain_loss(tensor_x, inject_tensor)
        return loss

    def model_fit(self):

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

                preds, z_mu, z, log_sigma2_c, probs, x_pro = self.model.infer_d_v_2(tensor_x, inject_tensor)
                z_ = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                Z[counter : counter + z.shape[0], :] = z_

        kmeans = KMeans(n_clusters=self.args.d_dim, n_init=20)
        predictions = kmeans.fit_predict(Z)
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        return predictions
