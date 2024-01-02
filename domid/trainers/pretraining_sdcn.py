import torch
import numpy as np
from sklearn.cluster import KMeans
from domid.dsets.make_graph_wsi import GraphConstructorWSI


class PretrainingSDCN:
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

    def pretrain_loss(self, tensor_x):
        return self.model.pretrain_loss(tensor_x)

    def kmeans_cluster_assignement(self):
        num_img = len(self.loader_tr.dataset)
        if self.args.task == "wsi" and self.args.aname == "sdcn":
            num_img = int(self.args.bs / 3)
        Z = np.zeros((num_img, self.model.zd_dim))
        counter = 0
        with (torch.no_grad()):
            for i, (tensor_x, vec_y, vec_d, *other_vars) in enumerate(self.loader_tr):

                tensor_x, vec_y, vec_d = (
                    tensor_x.to(self.device),
                    vec_y.to(self.device),
                    vec_d.to(self.device),
                )

                if self.args.task == "wsi" and self.args.aname == "sdcn":
                    patches_idx = self.model.random_ind[i]  # torch.randint(0, len(vec_y), (int(self.args.bs/3),))
                    tensor_x = tensor_x[patches_idx, :, :, :]
                    image_id = [image_id[patch_idx_num] for patch_idx_num in patches_idx]
                    adj_mx, spar_mx = GraphConstructorWSI().construct_graph(
                        tensor_x, image_id, self.model.graph_method, None
                    )
                    self.model.adj = spar_mx

                preds, z, probs, x_pro = self.model.infer_d_v_2(tensor_x)
                z_ = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                Z[counter : counter + z.shape[0], :] = z_

        kmeans = KMeans(n_clusters=self.args.d_dim, n_init=20)
        predictions = kmeans.fit_predict(Z)
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

    def model_fit(self):
        self.kmeans_cluster_assignement()
