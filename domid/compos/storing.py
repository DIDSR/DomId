import datetime
import os

import numpy as np
import pandas as pd
import torch
import pickle


class Storing():
    def __init__(self, args):
        self.args = args
        self.loss = []
        self.acc = []
        self.val_acc = []
        self.val_loss = []
        self.experiment_name = str(datetime.datetime.now())


    def storing(self, epoch, accuracy, loss, val_accuracy, val_loss):

        #arguments = [str(args.aname), str(args.model), str(args.prior), str(args.zd_dim), str(args.te_d), str(args.tr_d), str(args.L), str(args.lr), str(args.bs), str(args.pre_tr), str(args.warmup)]
        self.loss.append(loss)
        self.acc.append(accuracy)
        self.val_loss.append(val_loss.detach().cpu().numpy())
        self.val_acc.append(val_accuracy)

        if not os.path.exists("./notebooks/"+self.experiment_name):
            print('______Created directory to save result_________')
            os.mkdir("./notebooks/"+self.experiment_name)

        if epoch%5==0: # FIXME: hardcoded
            saving_dir = os.path.join("./notebooks",self.experiment_name)
            with open(saving_dir+"/training_loss.txt", 'w') as output:
                for row in self.loss:
                    output.write(str(row) + '\n')

            with open(saving_dir+"/training_accuracy.txt", 'w') as output:
                for row in self.acc:
                    output.write(str(row) + '\n')

            with open(saving_dir+"/testing_loss.txt", 'w') as output:
                for row in self.val_loss:
                    output.write(str(row) + '\n')

            with open(saving_dir+"/testing_accuracy.txt", 'w') as output:
                for row in self.val_acc:
                    output.write(str(row) + '\n')

            pickle.dump(self.args,open(saving_dir+'my_namespace.p','wb'))

        
    def saving_model(self, model):
        path_dict ="./notebooks/"+self.experiment_name+'/model_dict.pth'
        torch.save(model.state_dict(), path_dict)

    def storing_z_space(self, Z, predictions, vec_y_labels, vec_d_labels, image_id_labels):
        
        
        path ="./notebooks/"+self.experiment_name+"/Z_space.npy"
        np.save(path, Z)

        with open("./notebooks/"+self.experiment_name+"/vec_y_labels.txt", 'w') as output:
           
            for row in vec_y_labels:
                output.write(str(row) + '\n')
        with open("./notebooks/" + self.experiment_name + "/vec_d_labels.txt", 'w') as output:

            for row in vec_d_labels:
                output.write(str(row) + '\n')

        with open("./notebooks/"+self.experiment_name+"/predicted_labels.txt", 'w') as output:
            for row in predictions:
                output.write(str(row) + '\n')

        with open("./notebooks/" + self.experiment_name + "/img_id.txt", 'w') as output:

            for row in image_id_labels:
                output.write(str(row) + '\n')


