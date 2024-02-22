from domid.arg_parser import mk_parser_main
from domid.compos.exp.exp_main import Exp


def experiment_train(args):
    exp = Exp(args)
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    # exp.trainer.post_tr()


def test_VADE_CNN_nonbinary():
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
            "--epos",
            "2",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--prior",
            "Gaus",
            "--model_method",
            "cnn",
            "--pre_tr",
            "0",
            "--trainer",
            'cluster'
        ]
    )
    experiment_train(args)


def test_VADE_CNN():
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
            "--epos",
            "2",
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
            "--pre_tr",
            "0",
            "--trainer",
            'cluster'
        ]
    )
    experiment_train(args)


def test_VADE_nonbinary():
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
            "--epos",
            "2",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--prior",
            "Gaus",
            "--pre_tr",
            "0",
            "--trainer",
            'cluster'
        ]
    )
    experiment_train(args)


def test_VADE():
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
            "--epos",
            "2",
            "--bs",
            "2",
            "--split",
            "0.8",
            "--L",
            "5",
            "--debug",
            "--nocu",
            "--pre_tr",
            "0",
            "--trainer",
            'cluster'
        ]
    )
    experiment_train(args)
