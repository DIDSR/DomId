import os

import torch
import torch.utils.data
from domainlab.algos.observers.b_obvisitor import ObVisitor
from domainlab.models.model_diva import ModelDIVA
from domainlab.utils.utils_classif import mk_dummy_label_list_str
# from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
# from domainlab.compos.pcr.request import RequestVAEBuilderCHW
from domainlab.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix
from domainlab.algos.trainers.train_visitor import TrainerVisitor
from domainlab.compos.exp.exp_main import Exp

from domainlab.utils.test_img import mk_rand_xyd
from domainlab.utils.utils_classif import mk_dummy_label_list_str
from domid.algos.builder_vade_cnn import NodeAlgoBuilderVaDE
from domid.models.model_vade_cnn import ConvolutionalEncoder, ConvolutionalDecoder, ConvolutionalDecoder, ModelVaDECNN
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.models.model_vade import ModelVaDE
from domid.trainers.trainer_vade_pretraining import TrainerVADE
from domid.algos.observers.b_obvisitor_clustering_only import ObVisitorClusteringOnly
from domainlab.algos.msels.c_msel import MSelTrLoss
from domainlab.algos.msels.c_msel_oracle import MSelOracleVisitor
from domainlab.algos.observers.c_obvisitor_cleanup import ObVisitorCleanUp
from domid.compos.exp.exp_main import Exp
from domid.arg_parser import mk_parser_main
import os
def experiment_train(args):
    exp = Exp(args)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    #exp.trainer.post_tr()

def test_MNIST_train():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "7", "--tr_d", "0", "1",  "2",'--zd_dim', "5", "--d_dim", "3", "--dpath",
                              "zout", "--task", "mnist","--aname", "vade","--apath", "domid/algos/builder_vade.py",
                              "--bs", "2", "--split", "0.8", "--L", "5", "--debug", "--nocu"])
    experiment_train(args)
def test_MNISTcolor_train():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "7", "--tr_d", "0", "1", "2", '--zd_dim', "5", "--d_dim", "3", "--dpath",
                              "zout", "--task", "mnistcolor10", "--aname", "vade", "--apath",
                              "domid/algos/builder_vade.py",
                              "--bs", "2", "--split", "0.8", "--L", "5", "--debug", "--nocu"])
    experiment_train(args)

def test_MNIST_train_CNN():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "7", "--tr_d", "0", "1",  "2",'--zd_dim', "5", "--d_dim", "3", "--dpath",
                              "zout", "--task", "mnist","--aname", "vade","--apath", "domid/algos/builder_vade_cnn.py",
                              "--bs", "2", "--split", "0.8", "--L", "5", "--debug", "--nocu"])
    experiment_train(args)
def test_MNISTcolor_train_CNN():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "7", "--tr_d", "0", "1", "2", '--zd_dim', "5", "--d_dim", "3", "--dpath",
                              "zout", "--task", "mnistcolor10", "--aname", "vade", "--apath",
                              "domid/algos/builder_vade_cnn.py",
                              "--bs", "2", "--split", "0.8", "--L", "5", "--debug", "--nocu"])
    experiment_train(args)
def test_M2YD_train_MNISTcolor():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "7", "--tr_d", "0", "1",
                              '--zd_dim', "5", "--d_dim", "3", "--dpath",
                              "zout", "--task", "mnistcolor10", "--aname", "m2yd", "--apath",
                              "domid/algos/builder_m2yd.py",
                              "--bs", "2", "--split", "0.8", "--debug", "--nocu", "--gamma_y", "3500"])
    experiment_train(args)




