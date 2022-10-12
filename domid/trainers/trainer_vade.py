import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from domainlab.algos.trainers.a_trainer import TrainerClassif

from domid.trainers.pretraining import Pretraining
from domid.trainers.storing_plotting import Storing
from domid.utils.perf_cluster import PerfCluster


class TrainerVADE(TrainerClassif):
    def __init__(self, model, task, observer, device, writer, pretrain=True, aconf=None):
        """FIXME: add description of the parameters
        :param model: model to train
        :param task: task to train on
        :param observer: observer to notify
        :param device: device to use
        :param writer: tensorboard writer
        :param pretrain: whether to pretrain the model with MSE loss
        :param aconf: configuration parameters, including learning rate and pretrain threshold
        """
        super().__init__(model, task, observer, device, aconf)

        self.pretrain = pretrain
        self.pretraining_finished = not self.pretrain
        self.LR = aconf.lr
        self.warmup_beta = 0.01

        if not self.pretraining_finished:
            self.optimizer = optim.Adam(
                itertools.chain(self.model.encoder.parameters(), self.model.decoder.parameters()),
                lr=self.LR,
            )
            print("".join(["#"] * 60) + "\nPretraining initialized.\n" + "".join(["#"] * 60))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        self.epo_loss_tr = None
        self.writer = writer
        self.thres = aconf.pre_tr
        self.i_h, self.i_w = task.isize.h, task.isize.w
        self.args = aconf
        self.s = Storing(self.args)
        self.loader_val = task.loader_val

    def tr_epoch(self, epoch):
        """
        :param epoch: epoch number
        :return:
        """

        print("Epoch {}. ELBO loss".format(epoch)) if self.pretraining_finished else print("Epoch {}. MSE loss".format(epoch))
        self.model.train()
        self.epo_loss_tr = 0

        

        p = Pretraining(self.model, self.device, self.loader_tr, self.loader_val, self.i_h, self.i_w)
        acc_tr, _ = p.epoch_tr_acc()
        acc_val, _ = p.epoch_val_acc()



        if self.warmup_beta < 0.9 and self.pretraining_finished:
            self.warmup_beta = self.warmup_beta + 0.01

        for i, (tensor_x, vec_y, vec_d, *other_vars) in enumerate(self.loader_tr):
            if len(other_vars) > 0:
                machine, path = other_vars
            tensor_x, vec_y, vec_d = (
                tensor_x.to(self.device),
                vec_y.to(self.device),
                vec_d.to(self.device),
            )
            self.optimizer.zero_grad()

            if acc_val < self.thres and not self.pretraining_finished:
                loss = p.pretrain_loss(tensor_x)
            else:
                if not self.pretraining_finished:
                    self.pretraining_finished = True
                    # reset the optimizer
                    self.LR = self.LR/100
                    self.optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=self.LR,
                        betas=(0.5, 0.9),
                        weight_decay=0.000001,
                    )
                    
                    print("".join(["#"] * 60))
                    print("Epoch {}: Finished pretraining and starting to use ELBO loss.".format(epoch))
                    print("".join(["#"] * 60))

                loss = self.model.cal_loss(tensor_x, self.warmup_beta)
            
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.cpu().detach().item() #FIXME devide #  number of samples in the HER notebook 
            self.writer.add_scalar('learning rate', self.LR, epoch)
            self.writer.add_scalar('warmup', self.warmup_beta, epoch)
            if not self.pretraining_finished:
                self.writer.add_scalar('Pretraining', acc_tr, epoch)
                self.writer.add_scalar('Pretraining Loss', loss, epoch)
            else:
                self.writer.add_scalar('Training acc', acc_tr, epoch)
                self.writer.add_scalar('Loss', loss, epoch)

        preds, z_mu, z, _, _, x_pro = self.model.infer_d_v_2(tensor_x)
        name = "Output of the decoder" + str(epoch)
        imgs = torch.cat((tensor_x[0:8, :, :, :], x_pro[0:8, :, :, :],), 0)
        self.writer.add_images(name, imgs, epoch)

        if not self.pretraining_finished:
            gmm = p.GMM_fit()
        (preds_c,probs_c,z,z_mu,z_sigma2_log,mu_c,log_sigma2_c,pi,logits,) = self.model._inference(tensor_x)
        print("pi:")
        print(pi.cpu().detach().numpy())
        
        for i, (tensor_x, vec_y, vec_d, *other_vars) in enumerate(self.loader_val):
            if len(other_vars) > 0:
                machine, path = other_vars
            tensor_x, vec_y, vec_d = (
                tensor_x.to(self.device),
                vec_y.to(self.device),
                vec_d.to(self.device),
            )
            if acc_val < self.thres and not self.pretraining_finished:
                loss_val = p.pretrain_loss(tensor_x)
            else:
                loss_val = self.model.cal_loss(tensor_x, self.warmup_beta)

        self.s.storing(self.args, epoch, acc_tr, self.epo_loss_tr, acc_val, loss_val.sum())
        if epoch % 1 == 0:
            _, Z, domain_labels, machine_labels = p.prediction()
            self.s.storing_z_space(Z, domain_labels, machine_labels)

        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)  # FIXME change tr to te
        print("before training, model accuracy:", acc)

    def post_tr(self):
        print('training is done')
        self.observer.after_all()
