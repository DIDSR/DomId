import datetime
import os

from domainlab.algos.zoo_algos import AlgoBuilderChainNodeGetter
from domainlab.compos.exp.exp_utils import AggWriter

from domid.tasks.zoo_tasks import TaskChainNodeGetter

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
class Exp():
    """
    Exp is combination of Task, Algorithm, and Configuration (including random seed)
    """
    def __init__(self, args, task=None):
        """
        :param args:
        :param task:
        """
        self.task = task
        if task is None:
            self.task = TaskChainNodeGetter(args)()
        self.task.init_business(args)
        self.args = args
        self.visitor = AggWriter(self)
        algo_builder = AlgoBuilderChainNodeGetter(self.args)()  # request
        self.trainer = algo_builder.init_business(self)
        self.epochs = self.args.epos
        self.epoch_counter = 1


    def execute(self):
        """
        train model
        check performance by loading persisted model
        """
        self.args.lr = self.config['lr']
        print('LR in the experiment', self.args.lr)
        t_0 = datetime.datetime.now()
        print('\n Experiment start at :', str(t_0))
        t_c = t_0
        #self.trainer.before_tr()
        for epoch in range(1, self.epochs + 1):
            t_before_epoch = t_c
            flag_stop = self.trainer.tr_epoch(epoch)
            t_c = datetime.datetime.now()
            print("now: ", str(t_c), "epoch time: ", t_c - t_before_epoch, "used: ", t_c - t_0)
            # current time, time since experiment start, epoch time
            if flag_stop:
                self.epoch_counter = epoch
                break
            elif epoch == self.epochs:
                self.epoch_counter = self.epochs
            else:
                self.epoch_counter += 1
        print("Experiment finished at epoch:", self.epoch_counter,
              "with time:", t_c - t_0, "at", t_c)
        self.trainer.post_tr()

    def tuning(self):
        print(' checkpoint 1')
        self.config = {'lr': tune.loguniform(1e-4, 1e-1)}
        print(' checkpoint 2')
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=5,
            grace_period=1,
            reduction_factor=2)
        print(' checkpoint 3')
        self.reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["accuracy"])
        print(' checkpoint 4')
        self.result = tune.run(self.execute(),
            resources_per_trial={"cpu": 1, "gpu": 0},
            config=self.config,
            num_samples=2,
            scheduler=scheduler,
            progress_reporter=self.reporter)
        print('done')
