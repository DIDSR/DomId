import pandas as pd
import os
import numpy as np
class Storing():
    def __init__(self, args, epoch, accuracy):
        self.args = args
        self.epoch = epoch
        self.accuracy = accuracy

    def storing_plotting(self, args, epoch, accuracy):


        # Create the pandas DataFrame with column name is provided explicitly
        # constant lr for differentL: 0.0001
        # constant L for different lr: L =5
        # Constant L and lr for different zd_dim
        if not os.path.exists('results.cvs'):


            columns = ['L5_lr0.0001_z50','L10_lr0.0001_z50', 'L25_lr0.0001_z50',
                       'L5_lr0.00001_z50', 'L5_lr0.01_z50','L5_lr0.000001_z50',
                       'L5_lr0.0001_z10', 'L5_lr0.0001_z50', 'L5_lr0.0001_z100', 'L5_lr0.0001_z500']
            data = np.zeros((args.epos, 10))
            df = pd.DataFrame(data, columns = columns)

            experiment_name = 'L' + str(args.L) + '_lr' + str(args.lr) + '_z' + str(args.zd_dim)
            df.iloc[epoch][experiment_name] = accuracy
            df.to_csv('results.csv')
        else:
            df = pd.read_csv('results.csv')

            experiment_name = 'L' + str(args.L) + '_lr' + str(args.lr) + '_z' + str(args.zd_dim)
            df.iloc[epoch][experiment_name] = accuracy

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