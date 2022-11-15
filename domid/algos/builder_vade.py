import datetime

from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.utils.utils_cuda import get_device
from tensorboardX import SummaryWriter

from domid.algos.observers.b_obvisitor_clustering_only import \
    ObVisitorClusteringOnly
from domid.models.model_vade import ModelVaDE
from domid.trainers.trainer_vade import TrainerVADE


class NodeAlgoBuilderVaDE(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        Initialize model, observer, trainer. Return trainer.
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)

        zd_dim = args.zd_dim
        d_dim = args.d_dim
        L = args.L
        pretrain = args.pretrain
        now = 'zd_dim_'+str(zd_dim)+'_lr_'+str(args.lr)+'_'+str(datetime.datetime.now())
        model = ModelVaDE(zd_dim=zd_dim,
                          d_dim=d_dim,
                          device=device,
                          L=L,
                          i_c=task.isize.c,
                          i_h=task.isize.h,
                          i_w=task.isize.w,
                          args=args)
        observer = ObVisitorCleanUp(
            ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        writer = SummaryWriter(logdir="debug/"+now)
        trainer = TrainerVADE(model, task, observer, device, writer, pretrain = pretrain, aconf=args)

        return trainer


def get_node_na():
    return NodeAlgoBuilderVaDE
