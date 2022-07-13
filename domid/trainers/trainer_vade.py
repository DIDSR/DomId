"""
Base Class for trainer
"""
import abc
import torch
from domid.utils.perf_cluster import PerfCluster
from libdg.algos.trainers.a_trainer import TrainerClassif
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from domid.trainers.pretraining import pretraining



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
        counter = 0
        mse_n = 2
        # if epoch<10 (make a differnet file)
        #MSE loss - loss = self.MSE_cal_loss
        p = pretraining(self.model, self.device, self.optimizer, self.epo_loss_tr, self.loader_tr)
        for _, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            if epoch < mse_n:
                loss = p.pretrain_loss(tensor_x, mse_n, epoch)
            else:
                loss = self.model.cal_loss(tensor_x)

            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()
        if epoch < mse_n:
            p.GMM_fit()

            # if epoch == 10:
            #     #self.writer.add_embedding(tsne, metadata = meta, label_img=tensor_x)
            #     #self.writer.add_embedding(X, label_img=tensor_x)
            #     prediction, z_mu, z, log_sigma2_c, yita, x_pro = self.model.infer_d_v_2(tensor_x)
            #     #X = torch.flatten(z_mu, start_dim=1).cpu()
            #     # num = torch.argmax(vec_y, 1).cpu()
            #     # color = torch.argmax(vec_d, 1).cpu()
            #
            #     self.writer.add_embedding(z_mu, label_img=x_pro)
            #



            #print('Shapes for epcoh', counter, epoch, pred.shape, pi.shape, mu.shape, sigma.shape, yita.shape, x_pro.shape)
            counter += 1
        flag_stop = self.observer.update(epoch)  # notify observer




        #print(pred.shape, pi.shape, mu.shape, sigma.shape, yita.shape, x_pro.shape)
        #torch.Size([100, 7]) torch.Size([7]) torch.Size([7, 7]) torch.Size([100, 7])
        #print(pred[1, :], pi[1], sigma, yita[1, :])
        #print(sigma)

        #print(epoch, self.epo_loss_tr)
        self.writer.add_scalar('Trianing Loss', self.epo_loss_tr, epoch)
        # meta1 = ['1', '2', '3', '4', '5', '6', '7']
        # meta2 = ['11', '22', '33', '44', '55', '66', '77']

        pred, pi, mu, sigma, yita, x_pro = self.model.infer_d_v_2(tensor_x)
        if epoch ==1:
            name = "Input to the encoder" + str(epoch)
            self.writer.add_images(name, tensor_x, 0)

        name = "Input vs Output of the decoder"+str(epoch)
        imgs = torch.cat((tensor_x[0:8, :, :, :], x_pro[0:8, :, :, :],), 0)
        self.writer.add_images(name, imgs, 0)
        #self.writer.add_images(name, tensor_x[0:8, :, :, :], 0)
            # self.writer.add_image('Input epoch = 10 ', x_pro[2, :, :, :], 0)
            # self.writer.add_image('Input epoch = 10 ', x_pro[3, :, :, :], 0)

        return flag_stop

    def before_tr(self):
        """
        check the performance of randomly initialized weight
        """

        acc = PerfCluster.cal_acc(self.model, self.loader_tr, self.device)
        #print('ACC', acc)
        print("before training, model accuracy:", acc)

