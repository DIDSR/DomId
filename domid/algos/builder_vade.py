from libdg.algos.a_algo_builder import NodeAlgoBuilder
#from libdg.algos.trainers.train_basic import TrainerBasic
from domid.trainers.trainer_vade import TrainerVADE
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from libdg.utils.utils_cuda import get_device

from domid.algos.observers.b_obvisitor_clustering import ObVisitorClustering
from domid.models.model_vade import ModelVaDE

class NodeAlgoBuilderVaDE(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)
        # FIXME: add the nevessary function arguments:
        y_dim=len(task.list_str_y)
        #print('y dim in builder', y_dim)
        zd_dim = args.zd_dim
        #y_dim = len(task.list_str_y),



        model = ModelVaDE(y_dim=y_dim, zd_dim=zd_dim, device=device,  i_c = task.isize.c,
                          i_h = task.isize.h, i_w = task.isize.w, gamma_y = args.gamma_y,list_str_y = task.list_str_y, dim_feat_x = 10)
        observer = ObVisitorCleanUp(
            ObVisitorClustering(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))

        trainer = TrainerVADE(model, task, observer, device, aconf=args)

        return trainer


def get_node_na():
    return NodeAlgoBuilderVaDE
