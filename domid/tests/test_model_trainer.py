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
from domid.arg_parser import mk_parser_main
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

def test_train():
    parser = mk_parser_main()
    #args = parser.parse_args(["--te_d", "2", "--task", "mnist", "--debug"])
    #exp = Exp(args)
    args = parser.parse_args(["--te_d", "7", '--zd_dim', "5", "--d_dim", "1", "--dpath",
                              "zout", "--task", "mnist", "--split", "0.8", "--L", "5", "--debug"])
    #task = exp.task
    #args = exp.args
    device = 'cpu'

    zd_dim = args.zd_dim
    d_dim = args.d_dim
    L = args.L
    writer = None
    i_c, i_w, i_h = 3, 28, 28

    observer = None #ObVisitorClusteringOnly()#MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device))


    #parser = mk_parser_main()

    task = NodeTaskMNIST()
    #dset2 = task.get_dset_by_domain(args, 'digit2')
    #ldr = torch.utils.data.DataLoader(dset2[0])
    # device = 'cpu'
    # exp = Exp(args, task)
    # exp.execute()
    # observer = ObVisitorClusteringOnly(exp, MSelOracleVisitor(MSelTrLoss(max_es=args.es)), device)

    model= ModelVaDECNN(zd_dim=args.zd_dim, d_dim=args.d_dim, device=torch.device("cpu"), L=args.L,
                             i_c=i_c, i_w=i_w, i_h=i_h)
    try:
       TrainerVADE(model, task, observer, device, writer, aconf=args)
    except:
       pass

    TrainerVADE.tr_epoch(0)