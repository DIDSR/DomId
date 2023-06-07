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
        self.val_loss = []

        self.acc_y = []
        self.val_acc_y = []

        self.acc_d = []
        self.val_acc_d = []

        self.r_scores_tr = []
        self.r_scores_te = []
        model_name = str(args.aname)
        if args.dim_inject_y>0:
            model_name = 'cd'+str(args.aname)

        self.experiment_name = str(datetime.datetime.now()) + "_"  + str(args.task) + "_" + model_name
        self.last_epoch = args.epos


    def storing(self, epoch, acc_tr_y,acc_tr_d, loss_tr, acc_val_y, acc_val_d, loss_val, r_score_tr, r_score_te):
        """
        This function stores results for one epoch in csv files and command line arguments in the pkl file.

        :param epoch: epoch number
        :param acc_tr_y: training accuracy between predictions and vec_y labels for training set for one epoch
        :param acc_tr_d: training accuracy between predictions and vec_d labels for training set for one epoch
        :param loss_tr: training loss for one epoch
        :param acc_val_y: validation accuracy between predictions and vec_y labels for validation set for one epoch
        :param acc_val_d: validation accuracy between predictions and vec_d labels for validation set for one epoch
        :param loss_val: validation loss for one epoch
        :param r_score_tr: Pearson correlation coefficient between predictions and vec_y labels for training set for one epoch (only used for HER2)
        :param r_score_te: Pearson correlation coefficient between predictions and vec_y labels for test set for one epoch (only used for HER2)

        """

        self.loss.append(loss_tr)
        self.val_loss.append(loss_val)



        self.acc_y.append(acc_tr_y)
        self.val_acc_y.append(acc_val_y)

        self.acc_d.append(acc_tr_d)
        self.val_acc_d.append(acc_val_d)

        self.r_scores_tr.append(r_score_tr)
        self.r_scores_te.append(r_score_te)

        ex_path = "./notebooks/" + self.experiment_name
        if not os.path.exists("./notebooks/"+self.experiment_name):
            print('______Created directory to save result_________')

            os.mkdir(ex_path)
            df = pd.DataFrame(columns=['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss'])
            df.to_csv(os.path.join(ex_path, 'losses_accuracies.csv'), index=False)



        df = pd.read_csv(os.path.join(ex_path, 'losses_accuracies.csv'))
        saving_dir = os.path.join("./notebooks",self.experiment_name)
        loss_acc_df = pd.DataFrame({'epoch': epoch, 'loss': loss_tr,'accuracy': acc_tr_y,
                                    'val_loss': loss_val.item(), 'val_accuracy': acc_val_y}, index =[epoch] )
        df = pd.concat([df, loss_acc_df], join="inner", ignore_index=False)
        df.to_csv(os.path.join(ex_path, 'losses_accuracies.csv'), index=False)

        pickle.dump(self.args,open(os.path.join(saving_dir,'commandline_arguments.p'),'wb'))

        
    def saving_model(self, model):
        """
        This function saves the model in a pth file. This checkpoint is called every other epoch.
        """
        path_dict ="./notebooks/"+self.experiment_name+'/model_dict.pth'
        torch.save(model.state_dict(), path_dict)

    def storing_z_space(self, Z, predictions, vec_y_labels, vec_d_labels, image_id_labels):
        """
        This function stores the Z space and the predictions in a csv file after the last epoch.

        :param Z: Z space of the model
        :param predictions: predictions of the model in Z space
        :param vec_y_labels: vec_y labels corresponding to the predictions
        :param vec_d_labels: vec_d labels corresponding to the predictions
        :param image_id_labels: image_id labels corresponding to the predictions
        """
        
        exp_path =os.path.join("./notebooks/",self.experiment_name)

        np.save(os.path.join(exp_path, "Z_space.npy"), Z)
        pickle.dump(Z, open(os.path.join(exp_path, "Z_space_picle.p"), 'wb'))

        df = pd.DataFrame(columns=['vec_y_labels', 'vec_d_labels', 'predictions', 'image_id_labels'])

        df['vec_y_labels'] = vec_y_labels
        df['vec_d_labels'] = vec_d_labels
        df['predictions'] = predictions
        df['image_id_labels'] = image_id_labels


        df.to_csv(os.path.join(exp_path, 'clustering_results.csv'), index=False)
    def csv_dump(self, epoch):
        """
        This function stores the results of the experiment in a csv file which accumulates results for multiple
        experiments.
        """
        if os.path.exists(os.path.join(self.args.path_to_results, "results.csv")):
            results_df = pd.read_csv(os.path.join(self.args.path_to_results, "results.csv"))
        else:
            results_df = pd.DataFrame(columns=['dataset', 'model', 'seed', 'bs', 'zd_dim', 'lr', 'train_acc_y', 'test_acc_y',
                                               'train_acc_d', 'test_acc_d', 'R with scores train', 'R with scores test',

                                               'train_loss', 'test_loss','directory'])
            results_df.to_csv(os.path.join(self.args.path_to_results, "results.csv"), index=False)


        if self.args.inject_var:
            model_name = 'cd'+self.args.aname
        else:
            model_name = self.args.aname
        if self.last_epoch==epoch:
            row = [{'dataset': self.args.task, 'model': model_name,
                    'seed': self.args.seed, 'bs': self.args.bs,
                   'zd_dim': self.args.zd_dim,
                   'lr': self.args.lr,
                    'train_acc_y': self.acc_y[-1], 'test_acc_y': self.val_acc_y[-1],
                    'train_acc_d': self.acc_d[-1], 'test_acc_d': self.val_acc_d[-1],
                    'R with scores train': self.r_scores_tr[-1],
                    'R with scores test': self.r_scores_te[-1],
                    'train_loss': self.loss[-1],
                    'test_loss': self.val_loss[-1].item(), 'directory': self.experiment_name}]
            results_df = results_df.append(row, ignore_index=True)
            results_df.to_csv(os.path.join(self.args.path_to_results, "results.csv"), index=False)




