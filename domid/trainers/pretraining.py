import numpy as np
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture

from domid.utils.perf_cluster import PerfCluster


class Pretraining():
    def __init__(self, model, device, loader_tr, loader_val, i_h, i_w):
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

    def pretrain_loss(self, tensor_x, vec_y):
        """
        :param tensor_x: the input image
        :return: the loss
        """
        tensor_x = tensor_x.to(self.device)
        loss = self.model.pretrain_loss(tensor_x, vec_y)
        return loss

    def prediction(self):
        """
        This function is used for ease of storing the results. From training
        dataloader u=images, the prediction using currect state of the model
        are made.
        :return: tensor of dateset images
        :return: Z space of the current model
        :return: domain labels corresponding to Z space
        :return: machine (class labels) labels corresponding to Z space
        """
        num_img = len(self.loader_tr.dataset)
        Z = np.zeros((num_img, self.model.zd_dim))
        IMGS = np.zeros((num_img, 3, self.i_h, self.i_w))
        domain_labels = np.zeros((num_img, 1))
        machine_labels = []
        image_path = []
        counter = 0
        with torch.no_grad():
            for tensor_x, vec_y, vec_d, *other_vars in self.loader_tr:
                if len(other_vars) > 0:
                    machine, image_loc = other_vars
                    for i in range(len(machine)):
                        
                        machine_labels.append(machine[i])
                        image_path.append(image_loc[i])
                tensor_x = tensor_x.to(self.device)
                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x, vec_y)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                if z.shape[0]!=2:
                    IMGS[counter, :, :, :] = tensor_x.cpu().detach().numpy()
                    Z[counter, :] = z
                    preds = preds.detach().cpu()
                    domain_labels[counter, 0] = torch.argmax(preds, 1)+1
                    
                    
                else:
                    
                    IMGS[counter:counter+z.shape[0], :, :, :] = tensor_x.cpu().detach().numpy()
                    Z[counter:counter + z.shape[0], :] = z

                    # print(counter, counter+z.shape[0])
                    # print(domain_labels[0:50, 0])

                    preds = preds.detach().cpu()
                    domain_labels[counter:counter + z.shape[0], 0] = torch.argmax(preds, 1)+1
                counter+=z.shape[0]
                # print(counter)
        print(domain_labels)
        breakpoint()
        return IMGS, Z, domain_labels, machine_labels, image_path
    
    def prediction_te(self):
        """
        This function is used for ease of storing the results. From training
        dataloader u=images, the prediction using currect state of the model
        are made.
        :return: tensor of dateset images
        :return: Z space of the current model
        :return: domain labels corresponding to Z space
        :return: machine (class labels) labels corresponding to Z space
        """
        num_img = len(self.loader_val.dataset)
        Z = np.zeros((num_img, self.model.zd_dim))
        IMGS = np.zeros((num_img, 3, self.i_h, self.i_w))
        domain_labels = np.zeros((num_img, 1))
        machine_labels = []
        counter = 0
        with torch.no_grad():
            for tensor_x, vec_y, vec_d, *other_vars in self.loader_val:
                if len(other_vars) > 0:
                    machine, image_loc = other_vars
                    for i in range(len(machine)): 
                        machine_labels.append(machine[i][0])
                tensor_x = tensor_x.to(self.device)
                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x)
                z = z.detach().cpu().numpy()  # [batch_size, zd_dim]
                
                IMGS[counter:counter+z.shape[0], :, :, :] = tensor_x.cpu().detach().numpy()
                Z[counter:counter + z.shape[0], :] = z
                preds = preds.detach().cpu()
                domain_labels[counter:counter + z.shape[0], 0] = torch.argmax(preds, 1)+1
       


        return IMGS, Z, domain_labels, machine_labels 
    

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
            for tensor_x, vec_y, vec_d, *_ in self.loader_tr:
                tensor_x = tensor_x.to(self.device)
                preds, z_mu, z, *_ = self.model.infer_d_v_2(tensor_x)
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

    def epoch_tr_acc(self):
        acc, conf = PerfCluster.cal_acc(self.model, self.loader_tr, self.device, max_batches=None) 
        return acc, conf
    
    def epoch_val_acc(self):
        acc, conf = PerfCluster.cal_acc(self.model, self.loader_val, self.device, max_batches=None) 
        
        return acc, conf



