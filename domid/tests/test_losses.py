from domid.models.model_vade_cnn_nonbinary import ModelVaDECNN
from domid.models.model_vade_cnn import ModelVaDECNN
from domid.models.model_vade_nonbinary import ModelVaDE
from domid.models.model_vade_nonbinary import ModelVaDE
from domid.compos.exp.exp_main import Exp
from domid.arg_parser import mk_parser_main

def experiment_train(args):
    exp = Exp(args)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    # exp.trainer.post_tr()

def test_CNN_nonbinary():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "7", "--tr_d", "0", "1", "2", '--zd_dim', "5", "--d_dim", "3", "--dpath",
                              "zout", "--task", "mnistcolor10", "--aname", "vade", "--apath",
                              "domid/algos/builder_vade.py",
                              "--bs", "2", "--split", "0.8", "--L", "5", "--debug", "--nocu", "--nonbinary"])
    experiment_train(args)

def test_CNN():
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "7", "--tr_d", "0", "1", "2", '--zd_dim', "5", "--d_dim", "3", "--dpath",
                              "zout", "--task", "mnistcolor10", "--aname", "vade", "--apath",
                              "domid/algos/builder_vade.py",
                              "--bs", "2", "--split", "0.8", "--L", "5", "--debug", "--nocu"])
    experiment_train(args)

