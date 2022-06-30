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
        counter = 0
        for _, (tensor_x, vec_y, vec_d) in enumerate(self.loader_tr):
            tensor_x, vec_y, vec_d = \
                tensor_x.to(self.device), vec_y.to(self.device), vec_d.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.cal_loss(tensor_x, self.model.zd_dim)

            loss = loss.sum()
            #print(loss)

            #tsne = TSNE(n_components=2, random_state=0)

            #tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=500).fit_transform(X)

            #breakpoint()
            #labels = torch.cat((num, color))
            #labels = torch.cat((num.unsqueeze(0), color.unsqueeze(0)), 1)
            #
            # config = writer.ProjectorConfig()
            # embedding = config.embeddings.add()
            # embedding.tensor_name = embedding_var.name
            # embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
            # embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
            # embedding.sprite.single_image_dim.extend([28, 28])

            #meta = [str(int(num[i]))+str(int(color[i])) for i in range(len(color))]
            # meta.write('Index\tLabel\n')
            # for index, label in enumerate(labels):
            #     meta.write('{}\t{}\n'.format(index, label))

            if epoch == 150:
                #self.writer.add_embedding(tsne, metadata = meta, label_img=tensor_x)
                #self.writer.add_embedding(X, label_img=tensor_x)
                pred, pi, mu, sigma, yita, x_pro = self.model.infer_d_v_2(tensor_x)
                X = torch.flatten(x_pro, start_dim=1).cpu()
                num = torch.argmax(vec_y, 1).cpu()
                color = torch.argmax(vec_d, 1).cpu()




                self.writer.add_embedding(X, label_img=x_pro)

            #print('tsne', tsne.shape, num.shape, color.shape, meta.shape)
            #model.infer_d_v
            #writer.add_images()
            loss.backward()
            self.optimizer.step()
            self.epo_loss_tr += loss.detach().item()


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

        name = "Output of the decoder"+str(epoch)
        self.writer.add_images(name, x_pro, 0)
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

