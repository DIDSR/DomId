import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from domainlab.algos.trainers.a_trainer import TrainerClassif

from domid.utils.perf_cluster import PerfCluster
from domid.trainers.pretraining import Pretraining
from domid.trainers.storing_plotting import Storing

class TrainerVADE(TrainerClassif):
    def __init__(self, model, task, observer, device, writer, pretrain=True, aconf=None):
        """FIXME: add description of the parameters
        :param model:
        :param task:
        :param observer:
        :param device:
        :param writer:
        :param pretrain:
        :param aconf:
        """
        super().__init__(model, task, observer, device, aconf)

        self.pretrain = pretrain
        self.pretraining_finished = not self.pretrain
        self.LR = aconf.lr
        self.warmup_beta = 0.01

        if not self.pretraining_finished:
            self.optimizer = optim.Adam(
                itertools.chain(self.model.encoder.parameters(), self.model.decoder.parameters()), lr=self.LR
            )
            print("".join(["#"] * 60) + "\nPretraining initialized.\n" + "".join(["#"] * 60))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        self.epo_loss_tr = None
        self.writer = writer
        self.thres = aconf.pre_tr
        self.i_h, self.i_w = task.isize.h, task.isize.w
        self.args = aconf
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.95)
        # step_size = 10, gamma = 0.95
        # optimizer Adam
        self.s = Storing(self.args)


    def tr_epoch(self, epoch):
        """
        :param epoch: epoch number
        :return:
        """

        print("Epoch {}. ELBO loss".format(epoch)) if self.pretraining_finished else print("Epoch {}. MSE loss".format(epoch))
        self.model.train()
        self.epo_loss_tr = 0
        mse_n = 5  # FIXME: maybe have a command line argument to specify mse_n and elbo_n
        elbo_n = 100

        # if epoch>mse_n-2:
        #      self.LR = 0.001
        p = Pretraining(self.model, self.device, self.loader_tr, self.i_h, self.i_w)

        acc_d, _ = p.epoch_val_acc()
        print(acc_d)
        if self.warmup_beta <=1 and self.pretraining_finished:

            self.warmup_beta = self.warmup_beta * 2
        print('WARM UP', self.warmup_beta)

        for i, (tensor_x, vec_y, vec_d, *other_vars) in enumerate(self.loader_tr):

            if len(other_vars) > 0:
                machine, path = other_vars
            tensor_x, vec_y, vec_d = tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()




            if acc_d < self.thres and not self.pretraining_finished:
                loss = p.pretrain_loss(tensor_x, mse_n, epoch)
            else:
                if not self.pretraining_finished:
                    self.pretraining_finished = True
                    # reset the optimizer
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR, betas = (0.5, 0.9), weight_decay = 0.0001)
                    # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.95)
                    self.LR = self.LR/500

                    print("".join(["#"] * 60))
                    print("Epoch {}: Finished pretraining and starting to use ELBO loss.".format(epoch))
                    print("".join(["#"] * 60))


                loss = self.model.cal_loss(tensor_x, self.warmup_beta)


            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.cpu().detach().item()

        print("LEARNING RATE Value", self.LR)
        if not self.pretraining_finished:
            gmm = p.GMM_fit()
        preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self.model._inference(tensor_x)
        print("pi:")
        print(pi.cpu().detach().numpy())

        # ______________ changinge LR _____________________
        # if self.pretraining_finished:
        #     self.scheduler.step()
        #     print('learning rate', self.scheduler.get_lr()[0])
        # self.scheduler.step()
        #_______________________________________________________
        #
        # if not self.pretraining_finished:
        #     loss_store = 0
        # else:
        #     loss_store = self.epo_loss_tr
        # s = Storing(self.args,epoch,acc_d)
        # s.storing_plotting(self.args,epoch,acc_d,loss_store)


        # ___________ Tensorboard______________________________

        preds, z_mu, z, _, _, x_pro = self.model.infer_d_v_2(tensor_x)
        imgs = torch.cat((tensor_x[0:8, :, :, :], x_pro[0:8, :, :, :],), 0)
        print('mean constructed images', torch.mean((tensor_x[0, :, :, :])),'max reconstructed images', torch.max(x_pro[0, :, :, :]))
        name = "Decoder images epoch # " + str(epoch)
        self.writer.add_images(name, imgs, epoch)
        self.writer.add_scalar("Training acc", acc_d, epoch)

        if not self.pretraining_finished:
            self.writer.add_scalar("MSE loss", self.epo_loss_tr, epoch)
        else:
            self.writer.add_scalar("ELBO loss", self.epo_loss_tr, epoch)

        if epoch == 1:
            IMG, Z,  domain_labels, machine_labels = p.prediction()
            self.writer.add_embedding(Z, metadata=None, label_img=IMG, global_step=None, tag='default', metadata_header=None)
            self.s.storing_z_space(Z, domain_labels, machine_labels)
        #______________________STORIN__________________________________


        self.s.storing(self.args, epoch, acc_d, self.epo_loss_tr)


        flag_stop = self.observer.update(epoch)  # notify observer

        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """

        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)  # FIXME change tr to te
        print("before training, model accuracy:", acc)
