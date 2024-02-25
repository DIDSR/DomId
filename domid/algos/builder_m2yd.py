from domainlab.algos.a_algo_builder import NodeAlgoBuilder
#from domainlab.algos.msels.c_msel import MSelTrLoss
# from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.algos.trainers.train_basic import TrainerBasic
from domainlab.utils.utils_cuda import get_device

from domid.algos.observers.b_obvisitor_clustering import ObVisitorClustering
from domid.models.model_m2yd import mk_m2yd
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.utils.utils_cuda import get_device
from tensorboardX import SummaryWriter


from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domid.models.model_sdcn import mk_sdcn
#from domid.trainers.trainer_sdcn import TrainerCluster
from domid.trainers.zoo_trainer import TrainerChainNodeGetter



class NodeAlgoBuilderM2YD(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args)
        model = mk_m2yd()(
            y_dim=len(task.list_str_y),
            list_str_y=task.list_str_y,
            zd_dim=args.zd_dim,
            gamma_y=args.gamma_y,
            device=device,
            i_c=task.isize.c,
            i_h=task.isize.h,
            i_w=task.isize.w,
        )

        observer = ObVisitorCleanUp(
            ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelValPerf(max_es=args.es)), device))

        trainer = TrainerChainNodeGetter(args.trainer)()
        trainer.init_business(model, task, observer, device, args)

        return trainer, model, observer, device


def get_node_na():
    return NodeAlgoBuilderM2YD
