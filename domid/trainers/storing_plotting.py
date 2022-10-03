import datetime
import os

import numpy as np
import pandas as pd
import torch


class Storing():
    def __init__(self, args):
        self.args = args
        self.loss = []
        self.acc = []
        self.val_acc = []
        self.val_loss = []
        self.experiment_name = str(datetime.datetime.now())
        #self.epoch = epoch
        #self.accuracy = accuracy

    def storing(self, args, epoch, accuracy, loss, val_accuracy, val_loss):

        arguments = [str(args.aname), str(args.model), str(args.prior), str(args.zd_dim), str(args.te_d), str(args.tr_d), str(args.L), str(args.lr), str(args.bs), str(args.pre_tr), str(args.warmup)]
        self.loss.append(loss)
        self.acc.append(accuracy)
        self.val_loss.append(val_loss.detach().cpu().numpy())
        self.val_acc.append(val_accuracy)
        

        if not os.path.exists("./notebooks/"+self.experiment_name):
            print('______Created directory to save result_________')
            os.mkdir("./notebooks/"+self.experiment_name)

        if epoch%5==0:
            with open("./notebooks/"+self.experiment_name+"/training_loss.txt", 'w') as output:
                for row in self.loss:
                    output.write(str(row) + '\n')

            with open("./notebooks/"+self.experiment_name+"/training_accuracy.txt", 'w') as output:
                for row in self.acc:
                    output.write(str(row) + '\n')

            with open("./notebooks/"+self.experiment_name+"/testing_loss.txt", 'w') as output:
                for row in self.val_loss:
                    output.write(str(row) + '\n')

            with open("./notebooks/"+self.experiment_name+"/testing_accuracy.txt", 'w') as output:
                for row in self.val_acc:
                    output.write(str(row) + '\n')
                    
        if not os.path.exists("./notebooks/"+self.experiment_name+"/arguments.txt"):
            with open("./notebooks/"+self.experiment_name+"/arguments.txt", 'w') as output:
                for row in arguments:
                    output.write(row+'\n')
        


    def storing_z_space(self, Z, domain_labels, machine_labels):

        path ="./notebooks/"+self.experiment_name+"/Z_space.npy"
        np.save(path, Z)

        with open("./notebooks/"+self.experiment_name+"/domain_labels.txt", 'w') as output:
            for row in domain_labels:
                output.write(str(row) + '\n')

        with open("./notebooks/"+self.experiment_name+"/machine_labels.txt", 'w') as output:
            for row in machine_labels:
                output.write(str(row) + '\n')
        with open("./notebooks/"+self.experiment_name+"/times.txt", 'w') as output:
            times = [self.experiment_name, str(datetime.datetime.now())]
            for row in times:
                output.write(str(row)+'\n')




    def plot_histogram(self, epoch):
        if epoch > 0:
            IMGS, Z, domain_labels, machine_label = p.prediction()

            d1_machine = []
            d2_machine = []

            for i in range(0, len(domain_labels) - 1):

                if domain_labels[i] == 1:
                    d1_machine.append(machine_label[i])
                elif domain_labels[i] == 2:
                    d2_machine.append(machine_label[i])

            from matplotlib import pyplot as plt

            plt.subplot(2, 1, 1)
            plt.hist(d1_machine)
            plt.title('Class 1 Cancerous Tissue Scan Sources')

            plt.subplot(2, 1, 2)
            plt.hist(d2_machine)
            plt.title('Class 2 Cancerous Tissue Scan Sources')
            plt.show()
            # plt.savefig('./figures/hist_results'+str(epoch)+'.png')
            plt.close()
