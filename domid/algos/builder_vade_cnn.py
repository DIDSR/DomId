import datetime
from tensorboardX import SummaryWriter

from libdg.algos.a_algo_builder import NodeAlgoBuilder
from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from libdg.utils.utils_cuda import get_device

from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domid.models.model_vade_cnn import ModelVaDECNN
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

        now = datetime.datetime.now()

        model = ModelVaDECNN(
            zd_dim=zd_dim, d_dim=d_dim, device=device, i_c=task.isize.c, i_h=task.isize.h, i_w=task.isize.w
        )
        observer = ObVisitorCleanUp(ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        writer = SummaryWriter(logdir="debug_cnn/" + str(now))
        trainer = TrainerVADE(model, task, observer, device, writer, aconf=args)

        return trainer


def get_node_na():
    return NodeAlgoBuilderVaDE
