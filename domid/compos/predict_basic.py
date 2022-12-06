import torch
import numpy as np
from domid.utils.perf_cluster import PerfCluster

class Prediction():
    def __init__(self, model, device, loader_tr, loader_val, i_h, i_w, args):
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.model = model
        self.i_w = i_w
        self.i_h = i_h
        self.device = device
        self.args = args
        self.is_inject_domain = False
        if self.args.dim_inject_y > 0:
            self.is_inject_domain = True

    def mk_prediction(self):
        """
        This function is used for ease of storing the results. From training
        dataloader u=images, the prediction using currect state of the model
        are made.

        :return: tensor of dateset images
        :return: Z space of the current model
        :return: domain labels corresponding to Z space
        :return: machine (class labels) labels corresponding to Z space
        """

        num_img = len(self.loader_tr.dataset)-1
        Z = np.zeros((num_img, self.model.zd_dim))
        IMGS = np.zeros((num_img, 3, self.i_h, self.i_w))
        domain_labels = np.zeros((num_img, 1))
        machine_labels = []
        image_path = []
        counter = 0
        with torch.no_grad():
            for tensor_x, vec_y, vec_d, *other_vars in self.loader_tr:
                if len(other_vars) > 0:
                    machine, image_loc, pred_domain = other_vars

                    for i in range(len(machine)):
                        print(i)
                        machine_labels.append(machine[i])
                        image_path.append(image_loc[i])

                tensor_x = tensor_x.to(self.device)
                if self.is_inject_domain:
                    if vec_y.shape[1] + pred_domain.shape[1] == self.args.dim_inject_y and pred_domain.shape[1]!=0:
                        inject_tensor = torch.cat(vec_y, pred_domain)
                    elif vec_y.shape[1] == self.args.dim_inject_y:
                        inject_tensor = vec_y
                else:
                    inject_tensor = []


                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x, inject_tensor)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                if z.shape[0] == 1:
                    IMGS[counter, :, :, :] = tensor_x.cpu().detach().numpy()
                    Z[counter, :] = z
                    preds = preds.detach().cpu()
                    domain_labels[counter, 0] = torch.argmax(preds, 1) + 1

                else:

                    IMGS[counter:counter + z.shape[0], :, :, :] = tensor_x.cpu().detach().numpy()
                    Z[counter:counter + z.shape[0], :] = z

                    preds = preds.detach().cpu()
                    domain_labels[counter:counter + z.shape[0], 0] = torch.argmax(preds, 1) + 1
                counter += z.shape[0]

        return IMGS, Z, domain_labels, machine_labels, image_path

    def epoch_tr_acc(self):
        acc, conf = PerfCluster.cal_acc(self.model, self.loader_tr, self.device, max_batches=None)
        return acc, conf

    def epoch_val_acc(self):
        acc, conf = PerfCluster.cal_acc(self.model, self.loader_val, self.device, max_batches=None)

        return acc, conf

    # def prediction_te(self):
    #     """
    #     This function is used for ease of storing the results. From training
    #     dataloader u=images, the prediction using currect state of the model
    #     are made.
    #     :return: tensor of dateset images
    #     :return: Z space of the current model
    #     :return: domain labels corresponding to Z space
    #     :return: machine (class labels) labels corresponding to Z space
    #     """
    #     num_img = len(self.loader_val.dataset)
    #     Z = np.zeros((num_img, self.model.zd_dim))
    #     IMGS = np.zeros((num_img, 3, self.i_h, self.i_w))
    #     domain_labels = np.zeros((num_img, 1))
    #     machine_labels = []
    #     counter = 0
    #     with torch.no_grad():
    #         for tensor_x, vec_y, vec_d, *other_vars in self.loader_val:
    #             if len(other_vars) > 0:
    #                 machine, image_loc = other_vars
    #                 for i in range(len(machine)):
    #                     machine_labels.append(machine[i][0])
    #             tensor_x = tensor_x.to(self.device)
    #             preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x)
    #             z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
    #
    #             IMGS[counter:counter+z.shape[0], :, :, :] = tensor_x.cpu().detach().numpy()
    #             Z[counter:counter + z.shape[0], :] = z
    #             preds = preds.detach().cpu()
    #             domain_labels[counter:counter + z.shape[0], 0] = torch.argmax(preds, 1)+1
    #
    #
    #
    #     return IMGS, Z, domain_labels, machine_labels
    #
