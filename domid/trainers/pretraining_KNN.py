import numpy as np
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture

from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
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


    def model_fit(self):

        num_img = len(self.loader_tr.dataset)
        Z = np.zeros((num_img, self.model.zd_dim))
        X_pro = torch.zeros((num_img, 3*16*16))
        X = torch.zeros((num_img, 3*16*16))
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

                x_pro, z= self.model.inference_pretraining(tensor_x, inject_tensor)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                Z[counter:counter + z.shape[0], :] = z
                X_pro[counter:counter + z.shape[0], :] = x_pro.detach().cpu()
                X[counter:counter + z.shape[0], :] = tensor_x.detach().cpu().reshape(tensor_x.shape[0], -1)
                counter += z.shape[0]


        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        accuracy = cos_sim(X, X_pro)
        targets = torch.argmax(vec_y, dim=1).cpu().numpy()

        kmeans = KMeans(n_clusters=self.args.d_dim, n_init=20)

        predictions = kmeans.fit_predict(Z)

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)


        cost_matrix = np.zeros((self.args.d_dim, self.args.d_dim,))

        cost_matrix = cost_matrix - confusion_matrix(predictions, targets, labels=list(range(cost_matrix.shape[0])))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        conf_mat = (-1) * cost_matrix[:, col_ind]
        acc_d = np.diag(conf_mat).sum() / conf_mat.sum()
        print('K Means accuracy: ', acc_d)
        print('AE Cosine Similarity', accuracy.mean().item())

        return predictions
