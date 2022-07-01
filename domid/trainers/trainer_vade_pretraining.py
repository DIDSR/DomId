"""
Base Class for trainer
"""
import abc
import torch
from libdg.utils.perf import PerfClassif
from domid.utils.perf_cluster import PerfCluster
from libdg.algos.trainers.a_trainer import TrainerClassif
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE



class TrainerVADE(TrainerClassif):
    def __init__(self, model, task, observer, device, writer, aconf=None):
        super().__init__(model, task, observer, device, aconf)
        self.optimizer = optim.Adam(self.model.parameters(), lr=aconf.lr)
        self.epo_loss_tr = None
        self.writer = writer

    def tr_epoch(self, epoch):
        self.model.train()
        self.epo_loss_tr = 0
        #breakpoint()
        mse_n = 20
        elbo_n = 100
        for _, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            if epoch<mse_n:
                loss = self.model.pretrain_loss(tensor_x, self.model.zd_dim, self.device, epoch)
            else:
                loss = self.model.cal_loss(tensor_x, self.model.zd_dim)
            loss = loss.sum()

        loss.backward()
        self.optimizer.step()
        self.epo_loss_tr += loss.detach().item()

        _, z_mu, z, _, _, x_pro = self.model.infer_d_v_2(tensor_x)
        name = "Output of the decoder" + str(epoch)
        imgs = torch.cat((tensor_x[0:8,:, :, :], x_pro[0:8,:, :, :],), 0)
        self.writer.add_images(name, imgs, 0)

        if epoch<mse_n:
            self.writer.add_scalar('MSE loss', self.epo_loss_tr, epoch)
        else:
            self.writer.add_scalar('ELBO loss', self.epo_loss_tr, epoch)
        if epoch ==elbo_n:
            self.writer.add_embedding(z_mu, label_img=x_pro)



        flag_stop = self.observer.update(epoch)  # notify observer


        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """

        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)
        #print('ACC', acc)
        print("before training, model accuracy:", acc)

