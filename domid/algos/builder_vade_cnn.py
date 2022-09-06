import datetime
from tensorboardX import SummaryWriter

from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.utils.utils_cuda import get_device

from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domid.trainers.trainer_vade_pretraining import TrainerVADE  # CHANGE HERE


class NodeAlgoBuilderVaDE(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        device = get_device(args.nocu)

        zd_dim = args.zd_dim
        d_dim = args.d_dim
        lr = args.lr
        now = datetime.datetime.now()
        L = args.L

        if args.prior == "Gaus":
            from domid.models.model_vade_cnn_nonbinary import ModelVaDECNN
        else:
            from domid.models.model_vade_cnn import ModelVaDECNN

        model = ModelVaDECNN(
            zd_dim=zd_dim, d_dim=d_dim, device=device, L=L, i_c=task.isize.c, i_h=task.isize.h, i_w=task.isize.w
        )
        observer = ObVisitorCleanUp(ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        writer = SummaryWriter(logdir="CNN/" + str(now))
        trainer = TrainerVADE(model, task, observer, device, writer, aconf=args)

        return trainer


def get_node_na():
    return NodeAlgoBuilderVaDE
