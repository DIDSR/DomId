
import datetime
from tensorboardX import SummaryWriter

from domainlab.algos.a_algo_builder import NodeAlgoBuilder
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domainlab.utils.utils_cuda import get_device

from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domid.models.model_vade_cnn import ModelVaDECNN
from domid.trainers.trainer_vade_pretraining import TrainerVADE  # CHANGE HERE
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

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

        config = {
            "zd_dim": tune.choice([10, 50,100]),

        }
        zd_dim = config['zd_dim']
        scheduler = ASHAScheduler(
            max_t=10,
            grace_period=1,
            reduction_factor=2)


        model = ModelVaDECNN(
            zd_dim=zd_dim, d_dim=d_dim, device=device, L=L, i_c=task.isize.c, i_h=task.isize.h, i_w=task.isize.w
        )
        observer = ObVisitorCleanUp(ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))
        writer = SummaryWriter(logdir="CNN/" + str(now))

        trainer = TrainerVADE(model, task, observer, device, writer, aconf=args)
        result = tune.run(
            tune.with_parameters(trainer),
            resources_per_trial={"cpu": 1, "gpu": 1},
            config=config,
            metric="loss",
            mode="min",
            num_samples=1,
            scheduler=scheduler
        )
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))

        return trainer


def get_node_na():
    return NodeAlgoBuilderVaDE

