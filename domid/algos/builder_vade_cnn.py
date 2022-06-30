from libdg.algos.a_algo_builder import NodeAlgoBuilder
#from libdg.algos.trainers.train_basic import TrainerBasic
from domid.trainers.trainer_vade_pretraining import TrainerVADE #CHANGE HERE

from libdg.algos.msels.c_msel import MSelTrLoss
from libdg.algos.msels.c_msel_oracle import MSelOracleVisitor
from libdg.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from libdg.utils.utils_cuda import get_device
from tensorboardX import SummaryWriter
from domid.algos.observers.b_obvisitor_clustering import ObVisitorClustering
from domid.models.model_vade_cnn import ModelVaDECNN

class NodeAlgoBuilderVaDE(NodeAlgoBuilder):
    def init_business(self, exp):
        """
        return trainer, model, observer
        """
        task = exp.task
        args = exp.args
        #device = get_device(args.nocu)
        device = 'cpu'
        # FIXME: add the nevessary function arguments:
        y_dim=len(task.list_str_y)
        #print('y dim in builder', y_dim)
        zd_dim = args.zd_dim
        #y_dim = len(task.list_str_y),



        model = ModelVaDECNN(y_dim=y_dim, zd_dim=zd_dim, device=device,
                             i_h = task.isize.h, i_w = task.isize.w)
        observer = ObVisitorCleanUp(
            ObVisitorClustering(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        writer = SummaryWriter(logdir="debug_cnn")
        trainer = TrainerVADE(model, task, observer, device, writer,aconf=args)

        return trainer


def get_node_na():
    return NodeAlgoBuilderVaDE
