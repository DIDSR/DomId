import itertools

import torch
import torch.optim as optim
from domainlab.algos.trainers.a_trainer import AbstractTrainer

from domid.compos.predict_basic import Prediction
from domid.compos.storing import Storing
from domid.compos.tensorboard_fun import tensorboard_write
from domid.trainers.pretraining_GMM import Pretraining
from domid.utils.perf_cluster import PerfCluster


class TrainerCluster(AbstractTrainer):
    def __init__(self, model, task, observer, device, writer, pretrain=True, aconf=None):
        """
        :param model: model to train
        :param task: task to train on
        :param observer: observer to notify
        :param device: device to use
        :param writer: tensorboard writer
        :param pretrain: whether to pretrain the model with MSE loss
        :param aconf: configuration parameters, including learning rate and pretrain threshold
        """
        super().init_business(model, task, observer, device, aconf)

        self.pretrain = pretrain
        self.pretraining_finished = not self.pretrain
        self.lr = aconf.lr
        self.warmup_beta = 0.1
        if not self.pretraining_finished:
            self.optimizer = optim.Adam(
                itertools.chain(self.model.encoder.parameters(), self.model.decoder.parameters()), lr=self.lr
            )
            print("".join(["#"] * 60) + "\nPretraining initialized.\n" + "".join(["#"] * 60))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.epo_loss_tr = None
        self.writer = writer
        self.thres = aconf.pre_tr  # number of epochs for pretraining
        self.i_h, self.i_w = task.isize.h, task.isize.w
        self.args = aconf
        self.storage = Storing(self.args)
        self.loader_val = task.loader_val
        self.aname = aconf.aname


    def tr_epoch(self, epoch):
        """
        :param epoch: epoch number
        :return:
        """
        print("Epoch {}.".format(epoch)) if self.pretraining_finished else print("Epoch {}. Pretraining.".format(epoch))

        self.model.train()
        self.epo_loss_tr = 0

        pretrain = Pretraining(self.model, self.device, self.loader_tr, self.loader_val, self.i_h, self.i_w, self.args)
        prediction = Prediction(self.model, self.device, self.loader_tr, self.loader_val, self.i_h, self.i_w, self.args.bs)
        acc_tr, _ = prediction.epoch_tr_acc()
        acc_val, _ = prediction.epoch_val_acc()

        # ___________Define warm-up for ELBO loss_________
        if self.warmup_beta < 1 and self.pretraining_finished:
            self.warmup_beta = self.warmup_beta + 0.01

        # _____________one training epoch: start_______________________
        for i, (tensor_x, vec_y, vec_d, *other_vars) in enumerate(self.loader_tr):
            
            if len(other_vars) > 0:
                inject_tensor, image_id = other_vars
                if len(inject_tensor)>0:
                    inject_tensor = inject_tensor.to(self.device)



            tensor_x, vec_y, vec_d = (
                tensor_x.to(self.device),
                vec_y.to(self.device),
                vec_d.to(self.device),
            )
            self.optimizer.zero_grad()

            # __________________Pretrain/ELBO loss____________
            if epoch < self.thres and not self.pretraining_finished:
                loss = pretrain.pretrain_loss(tensor_x, inject_tensor)
            else:
                if not self.pretraining_finished:
                    self.pretraining_finished = True
                    # reset the optimizer

                    self.optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=self.lr,
                        betas=(0.5, 0.9),
                        weight_decay=0.0001,
                    )

                    print("".join(["#"] * 60))
                    print("Epoch {}: Finished pretraining and starting to use the full model loss.".format(epoch))
                    print("".join(["#"] * 60))

                loss = self.model.cal_loss(tensor_x, inject_tensor, self.warmup_beta)

            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.cpu().detach().item()
            # FIXME: devide #  number of samples in the HER notebook

        # after one epoch (all batches), GMM is calculated again and pi, mu_c
        # will get updated via this line.
        # name convention: mu_c is the mean for the Gaussian mixture cluster,
        # but mu alone means mean for decoded pixel
        if not self.pretraining_finished:
            pretrain.GMM_fit()

        # only z and pi needed
        (
            preds_c,
            probs_c,
            z,
            z_mu,
            z_sigma2_log,
            mu_c,
            log_sigma2_c,
            pi,
            logits,
        ) = self.model._inference(tensor_x)
        if self.aname =='vade':
            print("pi:")
            print(pi.cpu().detach().numpy())
        #__________________Validation_____________________
        for i, (tensor_x_val, vec_y_val, vec_d_val, *other_vars) in enumerate(self.loader_val):
            if len(other_vars) > 0:
                inject_tensor_val, img_id_val = other_vars
                if len(inject_tensor_val)>0:
                    inject_tensor_val = inject_tensor_val.to(self.device)
            tensor_x_val, vec_y_val, vec_d_val = (
                tensor_x_val.to(self.device),
                vec_y_val.to(self.device),
                vec_d_val.to(self.device),
            )

            if epoch < self.thres and not self.pretraining_finished:
                loss_val = pretrain.pretrain_loss(tensor_x_val, inject_tensor_val)
            else:
                loss_val = self.model.cal_loss(tensor_x_val, inject_tensor_val, self.warmup_beta)

        tensorboard_write(
            self.writer,
            self.model,
            epoch,
            self.lr,
            self.warmup_beta,
            acc_tr,
            loss,
            self.pretraining_finished,
            tensor_x,
            inject_tensor,
        )

        # _____storing results and Z space__________
        self.storage.storing(epoch, acc_tr, self.epo_loss_tr, acc_val, loss_val.sum())
        if epoch % 2 == 0:
            _, z_proj, predictions, vec_y_labels, vec_d_labels, image_id_labels = prediction.mk_prediction()
            #_, Z, domain_labels, machine_labels, image_locs = prediction.mk_prediction()

            self.storage.storing_z_space(z_proj, predictions, vec_y_labels,vec_d_labels, image_id_labels)
        if epoch % 10 == 0:
            self.storage.saving_model(self.model)
            

        flag_stop = self.observer.update(epoch)  # notify observer
        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)  # FIXME change tr to te
        print("before training, model accuracy:", acc)

    def post_tr(self):
        print("training is done")
        self.observer.after_all()
