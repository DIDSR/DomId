import datetime

from domainlab.algos.a_algo_builder import NodeAlgoBuilder
#from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_val import MSelValPerf
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.utils.utils_cuda import get_device
from tensorboardX import SummaryWriter

from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domid.models.model_dec import mk_dec
from domid.trainers.trainer_cluster import TrainerCluster

from domid.trainers.zoo_trainer import TrainerChainNodeGetter

class NodeAlgoBuilderDEC(NodeAlgoBuilder):
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
        pretrain = False
        if args.pre_tr > 0:
            pretrain = True

        now = "zd_dim_" + str(zd_dim) + "_lr_" + str(args.lr) + "_" + str(datetime.datetime.now())
        model = mk_dec()(
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

        #observer = ObVisitorCleanUp(ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        writer = SummaryWriter(logdir="debug/" + now)
        #
        trainer = TrainerChainNodeGetter(args.trainer)()
        trainer.init_business(model, task, observer, device, args)

        return trainer, model, observer, device


def get_node_na():
    return NodeAlgoBuilderDEC
