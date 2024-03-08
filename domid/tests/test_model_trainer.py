import os
import pytest
import shutil

from domid.algos.builder_vade import NodeAlgoBuilderVaDE
from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domid.arg_parser import mk_parser_main
from domid.compos.exp.exp_main import Exp
#from domid.models.model_vade import ModelVaDE
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.trainers.trainer_cluster import TrainerCluster
from domid.tests.utils import experiment_train


def train_MNISTcolor_AE(out_dir):
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
            "20",
            "--d_dim",
            "10",
            "--dpath",
            "zout",
            "--task",
            "mnistcolor10",
            "--model",
            "ae",
            "--apath",
            "domid/algos/builder_ae.py",
            "--bs",
            "600",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model_method",
            "cnn",
            "--prior",
            "Gaus",
            "--pre_tr",
            "1",
            "--epos",
            "3",
            "--trainer",
            "ae",
            "--feat_extract",
            "ae"

        ]
    )
    experiment_train(args, save_path=out_dir)


@pytest.fixture(scope="session")
def ae_weights(tmp_path_factory):
    # Create a temporary directory accessible by all tests
    ae_weights_dir = tmp_path_factory.mktemp("ae_weights_dir")
    # this will save the AE weights in that directory; note that the AE training is run only once, no matter how
    # often ae_weights() is used in the tests below.
    train_MNISTcolor_AE(ae_weights_dir)
    return ae_weights_dir

def test_MNIST_pretrain(tmp_path):
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
            "--model",
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
            "--model_method",
            "linear",
            "--prior",
            "Bern",
            "--pre_tr",
            "1",
            "--trainer",
            "cluster"
        ]
    )
    experiment_train(args, save_path=tmp_path)


def test_MNIST_train(tmp_path):
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
            "--model",
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
            "--model_method",
            "linear",
            "--prior",
            "Bern",
            "--pre_tr",
            "0",
            "--trainer",
            "cluster"
        ]
    )
    experiment_train(args, save_path=tmp_path)


def test_MNIST_train_CNN(tmp_path):
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
            "--model",
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
            "--model_method",
            "cnn",
            "--prior",
            "Bern",
            "--pre_tr",
            "0",
            "--trainer",
            "cluster"
        ]
    )
    experiment_train(args, save_path=tmp_path)


def test_MNISTcolor_train(tmp_path):
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
            "--model",
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
            "--model_method",
            "linear",
            "--prior",
            "Gaus",
            "--pre_tr",
            "0",
            "--trainer",
            "cluster"
        ]
    )

    experiment_train(args, save_path=tmp_path)


def test_MNISTcolor_train_CNN(tmp_path):

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
            "--model",
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
            "--model_method",
            "cnn",
            "--prior",
            "Gaus",
            "--pre_tr",
            "0",
            "--trainer",
            "cluster"
        ]
    )
    experiment_train(args, save_path=tmp_path)


def test_MNISTcolor_pretrain_CNN(tmp_path):
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
            "--model",
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
            "--model_method",
            "cnn",
            "--prior",
            "Gaus",
            "--pre_tr",
            "1",
            "--trainer",
            "cluster"
        ]
    )
    experiment_train(args, save_path=tmp_path)


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
            "--model",
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
            "--trainer",
            "basic"
        ]
    )
    experiment_train(args)


def test_MNIST_conditionalOne_train(tmp_path):
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
            "--model",
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
            "--model_method",
            "cnn",
            "--prior",
            "Gaus",
            "--pre_tr",
            "0",
            "--dim_inject_y",
            "10",
            "--inject_var",
            "digit",
            "--trainer",
            "cluster"
        ]
    )
    experiment_train(args, save_path=tmp_path)


def test_MNISTcolor_AE(ae_weights):
    # MNIST color cnn vade with pretraining
    assert os.path.exists(ae_weights)


def test_MNISTcolor_SDCN(tmp_path, ae_weights):
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
            "20",
            "--d_dim",
            "10",
            "--dpath",
            "zout",
            "--task",
            "mnistcolor10",
            "--model",
            "sdcn",
            "--apath",
            "domid/algos/builder_sdcn.py",
            "--bs",
            "10",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--model_method",
            "cnn",
            "--prior",
            "Gaus",
            "--pre_tr",
            "1",
            "--pre_tr_weight_path",
            str(ae_weights),
            "--epos",
            "3",
            "--trainer",
            "sdcn",
            "--feat_extract",
            "ae"
        ]
    )
    experiment_train(args, save_path=tmp_path)


# def test_MNIST_conditional_train():
#     # create a text file filled with 0s, 1s, and 2s
#     with open("domid/tests/domain_labels.txt", "w") as f:
#         for i in range(14897):
#             fake_label = random.randint(0, 3)
#             f.write(str(fake_label) + "\n")
#
#     parser = mk_parser_main()
#     args = parser.parse_args(
#         [
#             "--te_d",
#             "7",
#             "--tr_d",
#             "0",
#             "1",
#             "2",
#             "--zd_dim",
#             "5",
#             "--d_dim",
#             "3",
#             "--dpath",
#             "zout",
#             "--task",
#             "mnist",
#             "--model",
#             "vade",
#             "--apath",
#             "domid/algos/builder_vade.py",
#             "--bs",
#             "2",
#             "--split",
#             "0.8",
#             "--L",
#             "5",
#             "--debug",
#             "--nocu",
#             "--model",
#             "cnn",
#             "--prior",
#             "Gaus",
#             "--pre_tr",
#             "0",
#             "--dim_inject_y",
#             "13",
#             "--path_to_domain",
#             "domid/tests/",
#         ]
#     )
#     experiment_train(args)
#     # remove the file after the test
#     os.remove("domid/tests/domain_labels.txt")
