import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.optim as optim

from libdg.algos.trainers.a_trainer import TrainerClassif

from domid.utils.perf_cluster import PerfCluster
from domid.trainers.pretraining import Pretraining


class TrainerVADE(TrainerClassif):
    def __init__(self, model, task, observer, device, writer, aconf=None):
        super().__init__(model, task, observer, device, aconf)

        self.LR = aconf.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=aconf.lr)
        self.epo_loss_tr = None
        self.writer = writer

        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
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
        print('LEARNING RATE', self.LR)
        self.model.train()
        self.epo_loss_tr = 0
        #breakpoint()
        mse_n =5# FIXME: maybe have a command line argument to specify mse_n and elbo_n
        elbo_n = 300

        # if epoch>mse_n-2:
        #     self.LR = 0.0000001
        p = Pretraining(self.model, self.device, self.loader_tr)
        for i, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            if epoch<mse_n:
                loss = p.pretrain_loss(tensor_x, mse_n,epoch)
            else:
                loss = self.model.cal_loss(tensor_x)

            loss = loss.sum()
            # print("LOSS back", loss)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            self.epo_loss_tr += loss.cpu().detach().item()

        if epoch<mse_n:
            gmm = p.GMM_fit()
            self.model.log_pi.data = torch.from_numpy(np.log(gmm.weights_)).to(self.device).float()
            self.model.mu_c.data = torch.from_numpy(gmm.means_).to(self.device).float()
            self.model.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_)).to(self.device).float()

        import matplotlib.pyplot as plt
        preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = self.model._inference(tensor_x)
        z_mu = z_mu.detach().cpu().numpy()
        z_sigma2_log = z_sigma2_log.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        log_sigma2_c = log_sigma2_c.detach().cpu().numpy()
        pi = pi.detach().cpu().numpy()
        print('sum of weights', torch.sum(self.model.encoder.mu_layer.weight.data))
        mu_c = mu_c.detach().cpu().numpy()
        print('PARAMETERS', pi, mu_c, log_sigma2_c)
        plt.figure(dpi = 800)
        plt.subplot(6,1, 1)
        plt.imshow(z_mu)
        plt.title('Z mu', fontsize=8)
        plt.subplot(6,1, 2)


        plt.imshow(z_sigma2_log)
        plt.title('Z sigma2 log', fontsize=8)
        plt.subplot(6,1, 3)
        plt.imshow(z)
        plt.title('Z', fontsize=8)
        plt.subplot(6,1, 4)
        plt.imshow(mu_c)
        plt.title('Mu c', fontsize=8)
        plt.subplot(6,1, 5)
        plt.imshow(log_sigma2_c)
        plt.title('log sigma c', fontsize=8)
        plt.subplot(6,1, 6)
        plt.plot(pi)
        plt.title('pi', fontsize=8)
        plt.savefig('figures/'+str(epoch))

        # print('Shapes for epcoh', counter, epoch, pred.shape, pi.shape, mu.shape, sigma.shape, yita.shape, x_pro.shape)

        flag_stop = self.observer.update(epoch)  # notify observer

        preds, z_mu, z, _, _, x_pro = self.model.infer_d_v_2(tensor_x)
        name = "Output of the decoder" + str(epoch)
        imgs = torch.cat((tensor_x[0:8,:, :, :], x_pro[0:8,:, :, :],), 0)
        self.writer.add_images(name, imgs, 0)
        self.writer.add_scalar('Learning rate', self.LR, epoch)

        #
        if epoch<mse_n:
            self.writer.add_scalar('MSE loss', self.epo_loss_tr, epoch)
        else:
            self.writer.add_scalar('ELBO loss', self.epo_loss_tr, epoch)
            #self.writer.add_scalar('Reconstraction Accuracy (cos similarity)', reconstruction_acc, epoch)
            #self.writer.add_scalar('Domain clustering acc', clustering_acc, epoch)
        if epoch ==elbo_n:

            class_labels = torch.argmax(vec_y, 1)

            self.writer.add_embedding(z, metadata= class_labels, label_img=x_pro) #FIXME set global trainer step

        flag_stop = self.observer.update(epoch)  # notify observer

        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """
        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)
        print("before training, model accuracy:", acc)

