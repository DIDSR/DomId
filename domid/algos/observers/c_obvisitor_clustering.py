import abc
import os

import numpy as np
import torch
from domainlab.algos.observers.a_observer import AObVisitor
from domainlab.compos.exp.exp_utils import ExpModelPersistVisitor
from domainlab.tasks.task_folder_mk import NodeTaskFolderClassNaMismatch
from domainlab.utils.perf import PerfClassif
from domainlab.utils.utils_class import store_args


def pred2file(loader_te, model, device, fa='path_prediction.txt', flag_pred_scalar=False):
    """
    stores prediction to txt file
    """
    model.eval()
    model_local = model.to(device)
    for i, (x_s, y_s, *_, path) in enumerate(loader_te):
        x_s, y_s = x_s.to(device), y_s.to(device)
        pred, *_ = model_local.infer_y_vpicn(x_s)
        list_pred_list = pred.tolist()
        list_label_list = y_s.tolist()
        if flag_pred_scalar:
            list_pred_list = [np.asarray(pred).argmax() for pred in list_pred_list]
            list_label_list = [np.asarray(label).argmax() for label in list_label_list]
        list_pair_path_pred = list(zip(path, list_label_list, list_pred_list))  # label belongs to data
        with open(fa, 'a') as f:
            for pair in list_pair_path_pred:
                print(str(pair)[1:-1], file=f)  # 1:-1 removes brackets of tuple
    print("prediction saved in file ", fa)


class ObVisitor(AObVisitor):
    """
    Observer + Visitor pattern for model selection
    """
    @store_args
    def __init__(self, exp, model_sel, device):
        """
        observer trainer
        """
        self.host_trainer = None
        self.task = self.exp.task
        self.loader_te = self.exp.task.loader_te
        self.loader_tr = self.exp.task.loader_tr
        self.loader_val = self.exp.task.loader_val
        # Note loader_tr behaves/inherit different properties than loader_te
        self.epo_te = self.exp.args.epo_te
        self.epo = None
        self.acc_te = None
        self.keep_model = self.exp.args.keep_model

    def update(self, epoch):
        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            acc_tr_pool = PerfClassif.cal_acc(self.host_trainer.model, self.loader_tr, self.device)
            print("pooled train domain acc: ", acc_tr_pool)
            # test set has no domain label, so can be more custom
            acc_te = PerfClassif.cal_acc(self.host_trainer.model, self.loader_te, self.device)
            self.acc_te = acc_te
            print("out of domain test acc: ", acc_te)
        if self.model_sel.update():
            print("model selected")
            self.exp.visitor.save(self.host_trainer.model)
            print("persisted")
        return self.model_sel.if_stop()

    def accept(self, trainer):
        """
        accept invitation as a visitor
        """
        self.host_trainer = trainer
        self.model_sel.accept(trainer, self)

    def after_all(self):
        """
        After training is done
        """
        super().after_all()
        model_ld = self.exp.visitor.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()
        acc_te = PerfClassif.cal_acc(model_ld, self.loader_te, self.device)

        print("persisted model acc: ", acc_te)
        self.exp.visitor(acc_te)
        if isinstance(self.exp.task, NodeTaskFolderClassNaMismatch):
            pred2file(self.loader_te, self.host_trainer.model, self.device)

    def clean_up(self):
        """
        to be called by a decorator
        """

        print('was in clean up in c obvisitor, but did not clean anything')
        # if not self.keep_model:
        #     self.exp.visitor.remove("epoch")    # the last epoch
        #     # epoch exist to still have a model to evaluate if the training stops in between
        #     self.exp.visitor.remove("final")
        #     self.exp.visitor.remove()
        #     self.exp.visitor.remove("oracle")   # oracle means use out-of-domain test accuracy to select the model
