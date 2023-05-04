from domainlab.algos.observers.a_observer import AObVisitor

from domid.algos.observers.c_obvisitor_clustering import ObVisitor
from domid.utils.perf_cluster import PerfCluster


class ObVisitorClusteringOnly(ObVisitor):
    """
    Observer + Visitor pattern for clustering algorithms
    """

    def update(self, epoch):

        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            metric_tr, metric_te, r_score_tr, r_score_te = self.host_trainer.model.cal_perf_metric(
                self.loader_tr, self.device, self.loader_val) #note the loader is validation, not test dset
            self.metric_te = metric_te
            self.metric_tr = metric_tr


            print("pooled train clustering acc (vec_d correlation): ", metric_tr[2])
            print(metric_tr[3])

            print("clustering validation acc: ", metric_te[2])
            print(metric_te[3])

            print("pooled train clustering acc (vec_y correlation): ", metric_tr[0])
            print(metric_te[1])

            if r_score_tr is not None:
                print('Correlation with HER2 scores training', r_score_tr)
                print('Correlation with HER2 scores validation', r_score_te)



        self.exp.visitor.save(self.host_trainer.model)
        flag_stop = self.model_sel.if_stop()
        return flag_stop
    def accept(self, trainer):
        """
        accept invitation as a visitor
        """

        self.host_trainer = trainer
        self.perf_metric = self.host_trainer.model.create_perf_obj(self.task)
        self.model_sel.accept(trainer, self)


    def after_all(self):
        """
        After training is done
        """
        #super().after_all()
        model_ld = self.host_trainer.model  #self.exp.visitor.load()
        model_ld = model_ld.to(self.device)
        model_ld.eval()

        # Note that the final clustering performance is computed on the
        # validation set because the test set (loader_te) consists of different
        # (non-overlapping) clusters than training and validation sets.
        #acc_val, conf_mat_val = PerfCluster.cal_acc(model_ld, self.loader_val, self.device)
        #self.acc_val = acc_val
        #print("persisted model clustering acc: ", acc_val)
