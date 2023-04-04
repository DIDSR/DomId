import torch
import numpy as np
from domid.utils.perf_cluster import PerfCluster


class Prediction:
    def __init__(self, model, device, loader_tr, loader_val, i_h, i_w, args):
        self.loader_tr = loader_tr
        self.loader_val = loader_val
        self.model = model
        self.i_w = i_w
        self.i_h = i_h
        self.device = device
        self.args = args
        self.is_inject_domain = False
        # if self.args.dim_inject_y > 0:
        #     self.is_inject_domain = True

    def mk_prediction(self):
        """
        This function is used for ease of storing the results.
        Predictions are made for the training images using currect state of the model.

        :return: tensor of input dateset images
        :return: Z space representations of the input images through the current model
        :return: predicted domain/cluster labels
        :return: image acquisition machine labels for the input images (when applicable/available)
        """

        num_img = len(self.loader_tr.dataset)  # FIXME: this returns sample size + 1 for some reason
        z_proj = np.zeros((num_img, self.model.zd_dim))
        input_imgs = np.zeros((num_img, 3, self.i_h, self.i_w))

        image_id_labels = []
        vec_d_labels =[]
        vec_y_labels =[]
        predictions = []
        counter = 0
        #import pdb; pdb.set_trace()
        with torch.no_grad():

            for tensor_x, vec_y, vec_d, *other_vars in self.loader_tr:
                if len(other_vars) > 0:
                    inject_tensor, image_id = other_vars
                    if len(inject_tensor) > 0:
                        inject_tensor = inject_tensor.to(self.device)

                    for ii in range(0, self.args.bs):
                        vec_d_labels.append(torch.argmax(vec_d[ii, :]).item())
                        vec_y_labels.append(torch.argmax(vec_y[ii, :]).item())
                        image_id_labels.append(image_id[ii])




                tensor_x, vec_y, vec_d = (
                    tensor_x.to(self.device),
                    vec_y.to(self.device),
                    vec_d.to(self.device),
                )


                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x, inject_tensor)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                # if z.shape[0] == 1:
                #     input_imgs[counter, :, :, :] = tensor_x.cpu().detach().numpy()
                #     z_proj[counter, :] = z
                #     preds = preds.detach().cpu()
                #     predicted.append(torch.argmax(preds, 1).item()+1)
                #
                # else:
                    #pdb.set_trace()
                input_imgs[counter : counter + z.shape[0], :, :, :] = tensor_x.cpu().detach().numpy()
                z_proj[counter : counter + z.shape[0], :] = z

                preds = preds.detach().cpu()
                #domain_labels[counter : counter + z.shape[0], 0] = torch.argmax(preds, 1) + 1
                predictions+=(torch.argmax(preds, 1) + 1).tolist()
                counter += z.shape[0]

        return input_imgs, z_proj, predictions, vec_y_labels, vec_d_labels, image_id_labels

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
    #     input_imgs = np.zeros((num_img, 3, self.i_h, self.i_w))
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
    #             input_imgs[counter:counter+z.shape[0], :, :, :] = tensor_x.cpu().detach().numpy()
    #             Z[counter:counter + z.shape[0], :] = z
    #             preds = preds.detach().cpu()
    #             domain_labels[counter:counter + z.shape[0], 0] = torch.argmax(preds, 1)+1
    #
    #
    #
    #     return input_imgs, Z, domain_labels, machine_labels
    #
