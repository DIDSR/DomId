from domainlab.algos.observers.b_obvisitor import ObVisitor

from domid.utils.perf_cluster import PerfCluster

from domainlab.algos.observers.a_observer import AObVisitor
from domainlab.tasks.task_folder_mk import NodeTaskFolderClassNaMismatch
from domainlab.tasks.task_pathlist import NodeTaskPathListDummy
#class ObVisitorClustering(ObVisitor):
class ObVisitorClustering(AObVisitor):
    """
    Observer + Visitor pattern for clustering algorithms
    """

    def update(self, epoch):
        print("epoch:", epoch)
        # self.epo = epoch
        # if epoch % self.epo_te == 0:
        #     acc_tr_pool, conf_mat_tr = PerfCluster.cal_acc(self.host_trainer.model, self.loader_tr, self.device)
        #     print("pooled train clustering acc: ", acc_tr_pool)
        #     print(conf_mat_tr)
        #
        #     acc_val, conf_mat_val = PerfCluster.cal_acc(self.host_trainer.model, self.loader_val, self.device)
        #     self.acc_val = acc_val
        #     print("clustering validation acc: ", acc_val)
        #     print(conf_mat_val)
        breakpoint()
        self.epo = epoch
        if epoch % self.epo_te == 0:
            metric_te = self.host_trainer.model.cal_perf_metric(
                self.loader_tr, self.device, self.loader_te)
            self.metric_te = metric_te
        if self.model_sel.update():
            print("model selected")
            self.exp.visitor.save(self.host_trainer.model)
            print("persisted")
        flag_stop = self.model_sel.if_stop()
        # return flag_stop


        return super().update(epoch)

    def after_all(self):
        """
        After training is done
        """
        super().after_all()
        model_ld = self.exp.visitor.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()

        # Note that the final clustering performance is computed on the
        # validation set because the test set (loader_te) consists of different
        # (non-overlapping) clusters than training and validation sets.
        acc_val, conf_mat_val = PerfCluster.cal_acc(model_ld, self.loader_val, self.device)
        self.acc_val = acc_val
        print("persisted model clustering acc: ", acc_val)
