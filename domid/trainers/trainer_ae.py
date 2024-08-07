import torch.optim as optim
from domainlab.algos.trainers.a_trainer import AbstractTrainer
from tensorboardX import SummaryWriter

from domid.compos.predict_basic import Prediction
from domid.compos.tensorboard_fun import tensorboard_write
from domid.trainers.pretraining_KMeans import Pretraining
from domid.utils.perf_cluster import PerfCluster
from domid.utils.storing import Storing


class TrainerAE(AbstractTrainer):
    def init_business(self, model, task, observer, device, aconf, flag_accept=True):
        """
        :param model: model to train
        :param task: task to train on
        :param observer: observer to notify
        :param device: device to use
        :param writer: tensorboard writer
        :param pretrain: whether to pretrain the model with MSE loss
        :param aconf: configuration parameters, including learning rate and pretrain threshold
        """

        super().__init__()
        super().init_business(model, task, observer, device, aconf)
        print(model)
        if aconf.pre_tr > 0:
            self.pretrain = True
        else:
            self.pretrain = False

        self.pretraining_finished = not self.pretrain
        self.lr = aconf.lr
        self.warmup_beta = 0.1
        if not self.pretraining_finished:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            print("".join(["#"] * 60) + "\nPretraining initialized.\n" + "".join(["#"] * 60))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.epo_loss_tr = None
        self.thres = aconf.pre_tr  # number of epochs for pretraining
        self.i_h, self.i_w = task.isize.h, task.isize.w
        self.args = aconf
        self.storage = Storing(self.args)
        self.writer = SummaryWriter(logdir="debug/" + self.storage.experiment_name)
        self.loader_val = task.loader_tr
        self.aname = aconf.model

    def tr_epoch(self, epoch):
        """
        :param epoch: epoch number
        :return:
        """
        print("Epoch {}.".format(epoch)) if self.pretraining_finished else print("Epoch {}. Pretraining.".format(epoch))

        self.model.train()
        self.epo_loss_tr = 0

        pretrain = Pretraining(self.model, self.device, self.loader_tr, self.loader_val, self.i_h, self.i_w, self.args)

        prediction = Prediction(
            self.model, self.device, self.loader_tr, self.loader_val, self.i_h, self.i_w, self.args.bs
        )
        acc_tr_y, _, acc_tr_d, _ = prediction.epoch_tr_acc()
        acc_val_y, _, acc_val_d, _ = prediction.epoch_val_acc()
        r_score_tr = "None"
        r_score_te = "None"
        if self.args.task == "her2":
            r_score_tr = prediction.epoch_tr_correlation()
            r_score_te = prediction.epoch_val_correlation()  # validation set is used as a test set
        # ___________Define warm-up for ELBO loss_________
        if self.warmup_beta < 1 and self.pretraining_finished:
            self.warmup_beta = self.warmup_beta + 0.01
        # _____________one training epoch: start_______________________
        for i, (tensor_x, vec_y, vec_d, *other_vars) in enumerate(self.loader_tr):
            if i == 0:
                self.model.batch_zero = True

            if len(other_vars) > 0:
                inject_tensor, image_id = other_vars
                if len(inject_tensor) > 0:
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
                    # i.e., this is the first epoch after pre-training
                    # So we need to set the pretraining_finishend flag to True, and to reset the optimizer:
                    self.pretraining_finished = True
                    self.model.counter = 1
                    self.optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=self.lr,
                    )

                    print("".join(["#"] * 60))
                    print("Epoch {}: Finished pretraining and starting to use the full model loss.".format(epoch))
                    print("".join(["#"] * 60))

                loss = self.model.cal_loss(tensor_x, inject_tensor)

            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.cpu().detach().item()

        # after one epoch (all batches), GMM is calculated again and pi, mu_c
        # will get updated via this line.
        # name convention: mu_c is the mean for the Gaussian mixture cluster,
        # but mu alone means mean for decoded pixel

        if not self.pretraining_finished:
            pretrain.model_fit()

        # __________________Validation_____________________
        for i, (tensor_x_val, vec_y_val, vec_d_val, *other_vars) in enumerate(self.loader_val):
            if len(other_vars) > 0:
                inject_tensor_val, img_id_val = other_vars
                if len(inject_tensor_val) > 0:
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

        if self.writer != None:
            tensorboard_write(
                self.writer,
                self.model,
                epoch,
                self.lr,
                self.warmup_beta,
                acc_tr_y,
                loss,
                self.pretraining_finished,
                tensor_x,
                inject_tensor,
            )

        # _____storing results and Z space__________
        self.storage.storing(
            epoch, acc_tr_y, acc_tr_d, self.epo_loss_tr, acc_val_y, acc_val_d, loss_val, r_score_tr, r_score_te
        )
        if epoch % 2 == 0:
            _, z_proj, predictions, vec_y_labels, vec_d_labels, image_id_labels = prediction.mk_prediction()
            # _, Z, domain_labels, machine_labels, image_locs = prediction.mk_prediction()

            self.storage.storing_z_space(z_proj, predictions, vec_y_labels, vec_d_labels, image_id_labels)
        if epoch % 2 == 0:
            self.storage.saving_model(self.model)

        flag_stop = self.observer.update(epoch)  # notify observer
        # self.storage.csv_dump(epoch)
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
