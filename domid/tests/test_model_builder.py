import torch
import torch.utils.data
# from domainlab.algos.observers.b_obvisitor import ObVisitor
# from domainlab.models.model_diva import ModelDIVA
# from domainlab.utils.utils_classif import mk_dummy_label_list_str
#
# # from domainlab.compos.vae.utils_request_chain_builder import VAEChainNodeGetter
# # from domainlab.compos.pcr.request import RequestVAEBuilderCHW
# from domainlab.dsets.dset_poly_domains_mnist_color_default import DsetMNISTColorMix
# from domainlab.algos.trainers.train_visitor import TrainerVisitor
# from domainlab.compos.exp.exp_main import Exp
from domid.arg_parser import mk_parser_main
# from domainlab.utils.test_img import mk_rand_xyd
# from domainlab.utils.utils_classif import mk_dummy_label_list_str
#
# # from domid.algos.builder_vade_cnn import NodeAlgoBuilderVaDE
# # from domid.models.model_vade_cnn import ConvolutionalEncoder, ConvolutionalDecoder, ConvolutionalDecoder, ModelVaDECNN
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.models.model_vade import ModelVaDE
from domid.models.model_m2yd import ModelXY2D


def model_compiler(args, model):
    node = NodeTaskMNIST()
    dset2 = node.get_dset_by_domain(args, "digit2")
    ldr = torch.utils.data.DataLoader(dset2[0])

    for i, (tensor_x, vec_y, *_) in enumerate(ldr):
        if model.__class__.__name__ == "ModelVaDECNN" or model.__class__.__name__ == "ModelVaDE":
            preds_c, probs_c, z, z_mu, z_sigma2_log, mu_c, log_sigma2_c, pi, logits = model._inference(tensor_x)
            mu, log_sigma2 = model.encoder(tensor_x)
            model.decoder(z_mu)
            loss = model.cal_loss(tensor_x, 1)
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
            "cnn",
        ]
    )
    i_c, i_w, i_h = 3, 28, 28

    model = ModelVaDE(
        zd_dim=args.zd_dim, d_dim=args.d_dim, device=torch.device("cpu"), L=args.L, i_c=i_c, i_w=i_w, i_h=i_h, args=args
    )
    model_compiler(args, model)


def test_VaDE_linear():
    parser = mk_parser_main()
    args = parser.parse_args(
        ["--te_d", "7", "--zd_dim", "5", "--d_dim", "1", "--dpath", "zout", "--split", "0.8", "--L", "5"]
    )
    i_c, i_w, i_h = 3, 28, 28
    model = ModelVaDE(
        zd_dim=args.zd_dim, d_dim=args.d_dim, device=torch.device("cpu"), L=args.L, i_c=i_c, i_w=i_w, i_h=i_h, args=args
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
            "--aname",
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
    model = ModelXY2D(
        list_str_y=args.tr_d,
        y_dim=y_dim,
        zd_dim=args.zd_dim,
        gamma_y=args.gamma_y,
        device=torch.device("cpu"),
        i_c=3,
        i_h=28,
        i_w=28,
    )
    model_compiler(args, model)
