import datetime
import os
import pickle

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


    def storing(self, epoch, accuracy, loss, val_accuracy, val_loss):

        #arguments = [str(args.aname), str(args.model), str(args.prior), str(args.zd_dim), str(args.te_d), str(args.tr_d), str(args.L), str(args.lr), str(args.bs), str(args.pre_tr), str(args.warmup)]
        self.loss.append(loss)
        self.acc.append(accuracy)
        self.val_loss.append(val_loss.detach().cpu().numpy())
        self.val_acc.append(val_accuracy)
        ex_path = "./notebooks/" + self.experiment_name
        if not os.path.exists("./notebooks/"+self.experiment_name):
            print('______Created directory to save result_________')

            os.mkdir(ex_path)
            df = pd.DataFrame(columns=['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss'])
            df.to_csv(os.path.join(ex_path, 'losses_accuracies.csv'), index=False)



        df = pd.read_csv(os.path.join(ex_path, 'losses_accuracies.csv'))
        saving_dir = os.path.join("./notebooks",self.experiment_name)
        loss_acc_df = pd.DataFrame({'epoch': epoch, 'loss': loss,'accuracy': accuracy, 'val_loss': val_loss.item(), 'val_accuracy': val_accuracy}, index =[epoch] )
        df = pd.concat([df, loss_acc_df], join="inner", ignore_index=False)
        df.to_csv(os.path.join(ex_path, 'losses_accuracies.csv'), index=False)

        pickle.dump(self.args,open(os.path.join(saving_dir,'commandline_arguments.p'),'wb'))

        
    def saving_model(self, model):
        path_dict ="./notebooks/"+self.experiment_name+'/model_dict.pth'
        torch.save(model.state_dict(), path_dict)

    def storing_z_space(self, Z, predictions, vec_y_labels, vec_d_labels, image_id_labels):
        
        
        exp_path =os.path.join("./notebooks/",self.experiment_name)

        np.save(os.path.join(exp_path, "Z_space.npy"), Z)
        pickle.dump(Z, open(os.path.join(exp_path, "Z_space_picle.p"), 'wb'))

        df = pd.DataFrame(columns=['vec_y_labels', 'vec_d_labels', 'predictions', 'image_id_labels'])

        df['vec_y_labels'] = vec_y_labels
        df['vec_d_labels'] = vec_d_labels
        df['predictions'] = predictions
        df['image_id_labels'] = image_id_labels


        df.to_csv(os.path.join(exp_path, 'clustering_results.csv'), index=False)



