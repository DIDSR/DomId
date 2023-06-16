from domainlab.algos.observers.a_observer import AObVisitor

from domid.algos.observers.c_obvisitor_clustering import ObVisitor
from domid.utils.perf_cluster import PerfCluster

import torch
class ObVisitorClusteringOnly(ObVisitor):
    """
    Observer + Visitor pattern for clustering algorithms
    """

    def update(self, epoch):

        print("epoch:", epoch)
        self.epo = epoch
        if epoch % self.epo_te == 0:
            unbatched_tensor_x = torch.cat([tensor_x for tensor_x, _, _, *_ in self.loader_tr], dim=0).to(self.device)
            unbatched_vec_y = torch.cat([vec_y for _, vec_y, _, *_ in self.loader_tr], dim=0).to(self.device)
            unbatched_vec_d = torch.cat([vec_d for _, _, vec_d, *_ in self.loader_tr], dim=0).to(self.device)

            if len(next(iter(self.loader_tr))) > 3:
                try:
                    unbatched_inject_tensor = torch.cat(
                        [inject_tensor for _, _, _, inject_tensor, *_ in self.loader_tr], dim=0)
                except:
                    unbatched_inject_tensor = torch.Tensor([]).to(self.device)
                unbatched_image_id = torch.cat([image_id for _, _, _, _, image_id in self.loader_tr], dim=0).to(
                    self.device)


            metric_tr, metric_te, r_score_tr, r_score_te = self.host_trainer.model.cal_perf_metric(
                self.loader_tr, self.device, self.loader_tr) #note the loader is validation, not test dset
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
