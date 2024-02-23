import datetime

from domainlab.algos.a_algo_builder import NodeAlgoBuilder
#from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.utils.utils_cuda import get_device
from tensorboardX import SummaryWriter

from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domid.models.model_vade import mk_vade
from domid.trainers.trainer_cluster import TrainerCluster
from domid.trainers.zoo_trainer import TrainerChainNodeGetter

class NodeAlgoBuilderVaDE(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        Initialize model, observer, trainer. Return trainer.
        """
        task = exp.task
        args = exp.args

        device = get_device(args)

        zd_dim = args.zd_dim
        d_dim = args.d_dim
        L = args.L


        model = mk_vade()(
            zd_dim=zd_dim,
            d_dim=d_dim,
            device=device,
            L=L,
            i_c=task.isize.c,
            i_h=task.isize.h,
            i_w=task.isize.w,
            args=args,
        )

        observer = ObVisitorCleanUp(
            ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelValPerf(max_es=args.es)), device))

        trainer = TrainerChainNodeGetter(args.trainer)()
        trainer.init_business(model, task, observer, device, args)

        return trainer, model, observer, device


def get_node_na():
    return NodeAlgoBuilderVaDE
