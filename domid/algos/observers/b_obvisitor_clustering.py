from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.algos.observers.a_observer import AObVisitor
from domid.utils.perf_cluster import PerfCluster
from libdg.tasks.task_folder_mk import NodeTaskFolderClassNaMismatch
from libdg.compos.exp.exp_utils import ExpModelPersistVisitor
from libdg.algos.observers.b_obvisitor import ObVisitor
from libdg.utils.flows_gen_img_model import fun_gen

def pred2file(loader_te, model, device, fa='path_prediction.txt', flag_pred_scalar=False):
    model.eval()
    model_local = model.to(device)
    for i, (x_s, y_s, *_, path) in enumerate(loader_te):
        x_s, y_s = x_s.to(device), y_s.to(device)
        pred, *_ = model_local.infer_y_vpicn(x_s)
        # print(path)
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


class ObVisitorClustering(ObVisitor):
    """
    Observer + Visitor pattern for clustering algorithms
    """

    def update(self, epoch):
        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            acc_tr_pool, conf_mat_tr = PerfCluster.cal_acc(self.host_trainer.model, self.loader_tr, self.device)
            print("pooled train clustering acc: ", acc_tr_pool)
            print(conf_mat_tr)

            acc_val, conf_mat_val = PerfCluster.cal_acc(self.host_trainer.model, self.loader_val, self.device)
            self.acc_val = acc_val

            print("clustering validation acc: ", acc_val)
            print(conf_mat_val)
        return super().update(epoch)

    def after_all(self):
        """
        After training is done
        """
        #breakpoint()

        super().after_all()
        self.exp.visitor.save(self.host_trainer.model, "final")
        model_ld = self.exp.visitor.load()

        model_ld = model_ld.to(self.device)
        model_ld.eval()

        # Note that the final clustering performance is computed on the
        # validation set because the test set (loader_te) consists of different
        # (non-overlapping) clusters than training and validation sets.
        acc_val, conf_mat_val = PerfCluster.cal_acc(model_ld, self.loader_val, self.device)
        self.acc_val = acc_val
        print("persisted model clustering acc: ", acc_val)
        self.exp.visitor(acc_val)




    def clean_up(self):
        """
        to be called by a decorator
        """
        print('i was in clean up, but i didnt clean anything')
        #if not self.keep_model:
            #self.exp.visitor.remove("epoch")  # the last epoch
            # epoch exist to still have a model to evaluate if the training stops in between
            #self.exp.visitor.remove("final")
            #self.exp.visitor.remove()
            #self.exp.visitor.remove("oracle")

