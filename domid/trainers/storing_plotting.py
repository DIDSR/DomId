import pandas as pd
import os
import numpy as np
import torch
class Storing():
    def __init__(self, args):
        self.args = args
        self.loss = []
        self.acc = []
        #self.epoch = epoch
        #self.accuracy = accuracy

    def storing(self, args, epoch, accuracy, loss):


        # Create the pandas DataFrame with column name is provided explicitly
        # constant lr for different L: 0.0001
        # constant L for different lr: L =5
        # Constant L and lr for different zd_dim

        self.loss.append(loss)
        self.acc.append(accuracy)
        breakpoint()



        if epoch%5==0:
            with open("./notebooks/training_loss_p.txt", 'w') as output:
                for row in self.loss:
                    output.write(str(row) + '\n')

            with open("./notebooks/accuracy_p.txt", 'w') as output:
                for row in self.acc:
                    output.write(str(row) + '\n')
        # if epoch == 1:
        #
        #
        #     columns = ['L5_lr0.0005_z300']
        #     data = np.zeros((args.epos, len(columns)))
        #
        #     acc_df = pd.DataFrame(data, columns = columns)
        #     experiment_name = 'L' + str(args.L) + '_lr' + str(args.lr) + '_z' + str(args.zd_dim)
        #     acc_df.iloc[epoch].at[experiment_name] = accuracy
        #     acc_df.to_csv('./notebooks/results.csv')
        #
        #     loss_df = pd.DataFrame(data, columns=columns)
        #     experiment_name = 'L' + str(args.L) + '_lr' + str(args.lr) + '_z' + str(args.zd_dim)
        #     loss_df.iloc[epoch].at[experiment_name] = loss
        #     loss_df.to_csv('./notebooks/loss.csv')
        #
        # else:
        #
        #     acc_df = pd.read_csv('./notebooks/results.csv')
        #     loss_df = pd.read_csv('./notebooks/loss.csv')
        #     experiment_name = 'L' + str(args.L) + '_lr' + str(args.lr) + '_z' + str(args.zd_dim)
        #
        #     acc_df.iloc[epoch].at[experiment_name] = accuracy
        #     loss_df.iloc[epoch].at[experiment_name] = loss
        #     loss_df.to_csv('./notebooks/loss.csv')

    def storing_z_space(self, Z, domain_labels, machine_labels):

        with open('./notebooks/Z_space_p.npy', 'wb') as f:
            np.save(f, Z)

        with open("./notebooks/domain_labels_p.txt", 'w') as output:
            for row in domain_labels:
                output.write(str(row) + '\n')

        with open("./notebooks/machine_labels_p.txt", 'w') as output:
            for row in machine_labels:
                output.write(str(row) + '\n')




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