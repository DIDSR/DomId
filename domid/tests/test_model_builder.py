import torch
import torch.utils.data

from domid.arg_parser import mk_parser_main
#from domid.models.model_m2yd import ModelXY2D
from domid.models.model_vade import mk_vade
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.models.model_vade import mk_vade


def model_compiler(args, model):
    node = NodeTaskMNIST()
    dset2 = node.get_dset_by_domain(args, "digit2")
    ldr = torch.utils.data.DataLoader(dset2[0])

    for i, (tensor_x, vec_y, *_) in enumerate(ldr):
        if model.__class__.__name__ == "ModelVaDECNN" or model.__class__.__name__ == "ModelVaDE":
            (
                preds_c,
                probs_c,
                z,
                z_mu,
                z_sigma2_log,
                mu_c,
                log_sigma2_c,
                pi,
                logits,
            ) = model._inference(tensor_x)
            mu, log_sigma2 = model.encoder(tensor_x)
            model.decoder(z_mu)
            loss = model.cal_loss(x=tensor_x, inject_domain=[], warmup_beta=0.1)
        else:
            preds_c = model.infer_d_v(tensor_x)
            q_zd, zd_q, y_hat_logit = model.forward(tensor_x, vec_y)

            loss = model.cal_loss(tensor_x, y_hat_logit)

        if i > 5:
            break


def test_VaDE_CNN():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--zd_dim",
            "5",
            "--d_dim",
            "1",
            "--dpath",
            "zout",
            "--split",
            "0.8",
            "--L",
            "5",
            "--prior",
            "Bern",
            "--model",
            "vade",
            "--model_method",
            "cnn",
        ]
    )
    i_c, i_w, i_h = 3, 32, 32
    #zd_dim, d_dim, device, L, i_c, i_h, i_w, args
    model = mk_vade()(
        zd_dim=args.zd_dim,
        d_dim=args.d_dim,
        device=torch.device("cpu"),
        L=args.L,
        i_c=i_c,
        i_w=i_w,
        i_h=i_h,
        bs = args.bs,
        dim_inject_y = args.dim_inject_y,
        prior = args.prior,
        random_batching=args.random_batching,
        model_method=args.model_method,
        pre_tr_weight_path=args.pre_tr_weight_path,
        feat_extract=args.feat_extract
    )

    model_compiler(args, model)



def test_VaDE_linear():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--zd_dim",
            "5",
            "--d_dim",
            "1",
            "--dpath",
            "zout",
            "--split",
            "0.8",
            "--L",
            "5",
            "--model",
            "vade",
            "--model_method",
            "linear"
        ]
    )
    i_c, i_w, i_h = 3, 32, 32
    model = mk_vade()(
        zd_dim=args.zd_dim,
        d_dim=args.d_dim,
        device=torch.device("cpu"),
        L=args.L,
        i_c=i_c,
        i_w=i_w,
        i_h=i_h,
        bs = args.bs,
        dim_inject_y = args.dim_inject_y,
        prior = args.prior,
        random_batching=args.random_batching,
        model_method=args.model_method,
        pre_tr_weight_path=args.pre_tr_weight_path,
        feat_extract=args.feat_extract
    )

    model_compiler(args, model)


def test_m2yd():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "0",
            "1",
            "--tr_d",
            "3",
            "4",
            "--task",
            "mnistcolor10",
            "--model",
            "m2yd",
            "--d_dim",
            "2",
            "--apath=domid/algos/builder_m2yd.py",
            "--nocu",
            "--gamma_y",
            "3500",
        ]
    )
    y_dim = args.d_dim
    # model = ModelXY2D(
    #     list_str_y=args.tr_d,
    #     y_dim=y_dim,
    #     zd_dim=args.zd_dim,
    #     gamma_y=args.gamma_y,
    #     device=torch.device("cpu"),
    #     i_c=3,
    #     i_h=32,
    #     i_w=32,
    # )
    # model_compiler(args, model)
