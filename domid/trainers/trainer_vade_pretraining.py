import itertools
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.optim as optim

from domainlab.algos.trainers.a_trainer import TrainerClassif

from domid.utils.perf_cluster import PerfCluster
from domid.trainers.pretraining import Pretraining


class TrainerVADE(TrainerClassif):
    def __init__(self, model, task, observer, device, writer, pretrain=True, aconf=None):
        super().__init__(model, task, observer, device, aconf)

        self.pretrain = pretrain
        self.pretraining_finished = not self.pretrain
        self.LR = aconf.lr

        if not self.pretraining_finished:
            self.optimizer = optim.Adam(itertools.chain(self.model.encoder.parameters(), self.model.decoder.parameters()), lr=self.LR)
            print("".join(["#"]*60) + "\nPretraining initialized.\n" + "".join(["#"]*60))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        self.epo_loss_tr = None
        self.writer = writer
        self.thres = aconf.pre_tr



        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.95)
        # step_size = 10, gamma = 0.95
        #optimizer Adam

    def plot_loss_epoch(self, mse_loss, elbo_loss):
        x_mse = np.arrange(mse_n)
        x_elbo = np.arrange(elbo_loss)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(x_mse, mse_loss)
        plt.title('MSE loss')

        plt.subplot(2,1,2)
        plt.plot(x_elbo, elbo_loss)
        plt.title('ELBO loss')
        return plt.show()


    def tr_epoch(self, epoch):

        print("Epoch {}. ELBO loss".format(epoch)) if self.pretraining_finished else print("Epoch {}. MSE loss".format(epoch))
        #print('LEARNING RATE', self.LR)
        # print("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        #print(self.model.state_dict()['mu_c'])

        # Print optimizer's state_dict
        # print("Optimizer's state_dict:")
        # for var_name in self.optimizer.state_dict():
        #     print(var_name, "\t", self.optimizer.state_dict()[var_name])
        #
        # print()
        # breakpoint()
        self.model.train()
        self.epo_loss_tr = 0
        #breakpoint()
        mse_n =5# FIXME: maybe have a command line argument to specify mse_n and elbo_n
        elbo_n = 100

        # if epoch>mse_n-2:
        #      self.LR = 0.001
        p = Pretraining(self.model, self.device, self.loader_tr)

        acc_d, _ = p.epoch_val_acc()
        print(acc_d)


        for i, (tensor_x, vec_y, vec_d, machine, path) in enumerate(self.loader_tr):
            # import matplotlib.pyplot as plt
            # plt.imshow(tensor_x[1, :, :, :].reshape((100,100,3)))
            # plt.show()
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()


            if acc_d<self.thres:
                loss = p.pretrain_loss(tensor_x, mse_n,epoch)
            else:
                if not self.pretraining_finished:
                    self.pretraining_finished = True
                    # reset the optimizer
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
                    #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.95)
                    self.LR = self.LR*10
                    print("##############################################################")
                    print("Epoch {}: Finished pretraining and starting to use ELBO loss.".format(epoch))
                    print("##############################################################")

                #print('elbo LOSS______________________________')
                loss = self.model.cal_loss(tensor_x)
                self.thres = -1


                #print("PRINT LR", self.scheduler.print_lr())
            loss = loss.sum()
            # print("LOSS back", loss)
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.cpu().detach().item()
        print('LEARNING RATE Value', self.LR)
        if acc_d<self.thres:
            gmm = p.GMM_fit()
        preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self.model._inference(tensor_x)
        print("pi:")
        print(pi.cpu().detach().numpy())
        # if self.pretraining_finished == True:
        #     self.scheduler.step()
        #     print('learning rate', self.scheduler.get_lr()[0])


        #self.scheduler.step()

        preds, z_mu, z, _, _, x_pro = self.model.infer_d_v_2(tensor_x)
        imgs = torch.cat((tensor_x[0:8,:, :, :], x_pro[0:8,:, :, :],), 0)
        name = 'Decoder images epoch # ' + str(epoch)
        self.writer.add_images(name, imgs, epoch)
        self.writer.add_scalar('Training acc', acc_d, epoch)

        #
        if acc_d<self.thres:
            self.writer.add_scalar('MSE loss', self.epo_loss_tr, epoch)
        else:
            self.writer.add_scalar('ELBO loss', self.epo_loss_tr, epoch)
            #self.writer.add_scalar('Reconstraction Accuracy (cos similarity)', reconstruction_acc, epoch)
            #self.writer.add_scalar('Domain clustering acc', clustering_acc, epoch)
        if epoch==1:

            IMGS, Z, domain_labels, machine_label = p.prediction()
            class_labels = torch.argmax(vec_y[1:], 1)

            print('before writer', z[1:, :].shape, vec_y[1:, :].shape)
            self.writer.add_embedding(Z, metadata=domain_labels ,label_img=IMGS, global_step = epoch, tag = str(epoch)+'_'+str(acc_d)) #FIXME set global trainer step

        flag_stop = self.observer.update(epoch)  # notify observer

        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """

        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device) #FIXME change tr to te
        print("before training, model accuracy:", acc)

