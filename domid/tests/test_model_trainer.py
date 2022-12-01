import os
import random

from domid.algos.builder_vade import NodeAlgoBuilderVaDE
from domid.algos.observers.b_obvisitor_clustering_only import \
    ObVisitorClusteringOnly
from domid.arg_parser import mk_parser_main
from domid.compos.exp.exp_main import Exp
from domid.models.model_vade import ModelVaDE
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.trainers.trainer_vade import TrainerVADE


def experiment_train(args):
    exp = Exp(args)
    # exp.execute()
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    # exp.trainer.post_tr()


def test_MNIST_pretrain():
    # MNIST vade linear test for pretaining
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnist",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "linear",
            "--prior",
            "Bern",
            "--pretrain",
            "1",
        ]
    )
    experiment_train(args)


def test_MNIST_train():
    # MNIST vade linear test without pretaining
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnist",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "linear",
            "--prior",
            "Bern",
            "--pretrain",
            "0",
        ]
    )
    experiment_train(args)


def test_MNIST_train_CNN():
    # MNIST vade CNN without pretraining
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnist",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "cnn",
            "--prior",
            "Bern",
            "--pretrain",
            "0",
        ]
    )
    experiment_train(args)


def test_MNISTcolor_train():
    # MNIST color linear vade without pretraining
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnistcolor10",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "linear",
            "--prior",
            "Gaus",
            "--pretrain",
            "0",
        ]
    )
    experiment_train(args)


def test_MNISTcolor_train_CNN():

    # MNIST color cnn vade without pretraining
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnistcolor10",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "cnn",
            "--prior",
            "Gaus",
            "--pretrain",
            "0",
        ]
    )
    experiment_train(args)


def test_MNISTcolor_pretrain_CNN():
    # MNIST color cnn vade with pretraining
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnistcolor10",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "cnn",
            "--prior",
            "Gaus",
            "--pretrain",
            "1",
        ]
    )
    experiment_train(args)


def test_M2YD_train_MNISTcolor():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "0",
            "2",
            "1",
            "--tr_d",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "--task",
            "mnistcolor10",
            "--aname",
            "m2yd",
            "--zd_dim",
            "7",
            "--apath",
            "domid/algos/builder_m2yd.py",
            "--epos",
            "2",
            "--bs",
            "2",
            "--debug",
            "--nocu",
            "--gamma_y",
            "3500",
        ]
    )
    experiment_train(args)


def test_MNISTcolor_conditionalOne_train():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnist",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "cnn",
            "--prior",
            "Gaus",
            "--pretrain",
            "0",
            "--dim_inject_y",
            "10",
        ]
    )
    experiment_train(args)


def test_MNISTcolor_conditionalTwo_train():
    # create a text file filled with 0s, 1s, and 2s
    with open("domid/tests/domain_labels.txt", "w") as f:
        for i in range(14897):
            fake_label = random.randint(0, 3)
            f.write(str(fake_label) + "\n")

    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--zd_dim",
            "5",
            "--d_dim",
            "3",
            "--dpath",
            "zout",
            "--task",
            "mnist",
            "--aname",
            "vade",
            "--apath",
            "domid/algos/builder_vade.py",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model",
            "cnn",
            "--prior",
            "Gaus",
            "--pretrain",
            "0",
            "--dim_inject_y",
            "10",
            "--path_to_domain",
            "domid/tests/",
        ]
    )
    experiment_train(args)
    # remove the file after the test
    os.remove("domid/tests/domain_labels.txt")
