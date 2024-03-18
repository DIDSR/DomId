import numpy as np
import torch

from domid.dsets.make_graph_wsi import GraphConstructorWSI
from domid.utils.perf_cluster import PerfCluster
from domid.utils.perf_similarity import PerfCorrelationHER2


class Prediction:
    def __init__(self, model, device, loader_tr, loader_val, i_h, i_w, bs):
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.model = model
        self.i_w = i_w
        self.i_h = i_h
        self.device = device

    def mk_prediction(self):
        """
        This function is used for ease of storing the results.
        Predictions are made for the training images using currect state of the model.

        :return: tensor of input dateset images
        :return: Z space representations of the input images through the current model
        :return: predicted domain/cluster labels
        :return: image acquisition machine labels for the input images (when applicable/available)
        """

        num_img = len(self.loader_tr.dataset)

        if self.model.random_batching:
            bs = next(iter(self.loader_tr))[0].shape[0]
            num_img = int(bs / 3 * num_img)
        z_proj = np.zeros((num_img, self.model.zd_dim))
        prob_proj = np.zeros((num_img, self.model.d_dim))
        input_imgs = np.zeros((num_img, 3, self.i_h, self.i_w))

        image_id_labels = []
        vec_d_labels = []
        vec_y_labels = []
        predictions = []
        counter = 0

        with torch.no_grad():

            for i, (tensor_x, vec_y, vec_d, *other_vars) in enumerate(self.loader_tr):
                if len(other_vars) > 0:
                    inject_tensor, image_id = other_vars
                    if len(inject_tensor) > 0:
                        inject_tensor = inject_tensor.to(self.device)

                if self.model.random_batching:
                    patches_idx = self.model.random_ind[i]  # torch.randint(0, len(vec_y), (int(self.args.bs/3),))
                    tensor_x = tensor_x[patches_idx, :, :, :]
                    vec_y = vec_y[patches_idx, :]
                    vec_d = vec_d[patches_idx, :]
                    image_id = [image_id[patch_idx_num] for patch_idx_num in patches_idx]
                    adj_mx, spar_mx = GraphConstructorWSI(self.model.graph_method).construct_graph(
                        tensor_x, image_id, None
                    )
                    self.model.adj = spar_mx

                for ii in range(0, tensor_x.shape[0]):

                    vec_d_labels.append(torch.argmax(vec_d[ii, :]).item())
                    vec_y_labels.append(torch.argmax(vec_y[ii, :]).item())
                    image_id_labels.append(image_id[ii])

                tensor_x, vec_y, vec_d = (
                    tensor_x.to(self.device),
                    vec_y.to(self.device),
                    vec_d.to(self.device),
                )

                if self.model.model != "sdcn":
                    results = self.model.infer_d_v_2(tensor_x, inject_tensor)
                else:
                    results = self.model.infer_d_v_2(tensor_x)
                preds, z, probs, x_pro = results[0], results[1], results[-2], results[-1]
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                input_imgs[counter : counter + tensor_x.shape[0], :, :, :] = tensor_x.cpu().detach().numpy()
                z_proj[counter : counter + tensor_x.shape[0], :] = z
                prob_proj[counter : counter + tensor_x.shape[0], :] = probs

                preds = preds.detach().cpu()
                # domain_labels[counter : counter + z.shape[0], 0] = torch.argmax(preds, 1) + 1
                predictions += (torch.argmax(preds, 1) + 1).tolist()
                counter += tensor_x.shape[0]

        return input_imgs, z_proj, predictions, vec_y_labels, vec_d_labels, image_id_labels

    def epoch_tr_acc(self):
        """
        This function used to calculate accuracy and confusion matrix for training set for both vec_d and vec_y labels and predictions.
        """
        # hungarian_acc_y_s, conf_mat_y_s, hungarian_acc_d_s, conf_mat_d_s
        acc_vec_y, conf_y, acc_vec_d, conf_d = PerfCluster.cal_acc(
            self.model, self.loader_tr, self.device, max_batches=None
        )
        return acc_vec_y, conf_y, acc_vec_d, conf_d

    def epoch_val_acc(self):
        """
        This function used to calculate accuracy and confusion matrix for validation set for both vec_d and vec_y labels and predictions.
        """
        acc_vec_y, conf_y, acc_vec_d, conf_d = PerfCluster.cal_acc(
            self.model, self.loader_val, self.device, max_batches=None
        )

        return acc_vec_y, conf_y, acc_vec_d, conf_d

    def epoch_tr_correlation(self):
        """
        This function used to calculate correlation with HER2 scores for training set. Only used for HER2 dataset/task.
        """

        correlation_tr = PerfCorrelationHER2.cal_acc(self.model, self.loader_tr, self.device, max_batches=None)
        return correlation_tr

    def epoch_val_correlation(self):
        """
        This function used to calculate correlation with HER2 scores for valiation set. Only used for HER2 dataset/task.
        """
        correlation_val = PerfCorrelationHER2.cal_acc(self.model, self.loader_val, self.device, max_batches=None)
        return correlation_val
