"""
Base Class for trainer
"""
import abc
import torch
from domid.utils.perf_cluster import PerfCluster
from domainlab.algos.trainers.a_trainer import TrainerClassif
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from domid.trainers.pretraining import Pretraining



class TrainerVADE(TrainerClassif):
    def __init__(self, model, task, observer, device, writer, aconf=None):
        super().__init__(model, task, observer, device, aconf)
        self.optimizer = optim.Adam(self.model.parameters(), lr=aconf.lr)
        self.epo_loss_tr = None
        self.writer = writer

    def tr_epoch(self, epoch):
        self.model.train()
        self.epo_loss_tr = 0
        for k, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()

            loss = self.model.cal_loss(tensor_x)

            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()


        if self.writer is not None:
            self.writer.add_scalar('Trianing Loss', self.epo_loss_tr, epoch)

            pred, pi, mu, sigma, yita, x_pro = self.model.infer_d_v_2(tensor_x)
            if epoch ==1:
                name = "Input to the encoder" + str(epoch)
                self.writer.add_images(name, tensor_x, 0)

            name = "Input vs Output of the decoder"+str(epoch)
            imgs = torch.cat((tensor_x[0:8, :, :, :], x_pro[0:8, :, :, :],), 0)
            self.writer.add_images(name, imgs, 0)


        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """

        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)
        #print('ACC', acc)
        print("before training, model accuracy:", acc)

