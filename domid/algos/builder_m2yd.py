from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.utils.utils_cuda import get_device

from domid.algos.observers.b_obvisitor_clustering import ObVisitorClustering
from domid.models.model_m2yd import ModelXY2D


class NodeAlgoBuilderM2YD(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args)
        model = ModelXY2D(
            y_dim=len(task.list_str_y),
            list_str_y=task.list_str_y,
            zd_dim=args.zd_dim,
            gamma_y=args.gamma_y,
            device=device,
            i_c=task.isize.c,
            i_h=task.isize.h,
            i_w=task.isize.w,
        )
        observer = ObVisitorCleanUp(ObVisitorClustering(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        trainer = TrainerBasic()
        trainer.init_business(model, task, observer, device, args)
        return trainer


def get_node_na():
    return NodeAlgoBuilderM2YD
