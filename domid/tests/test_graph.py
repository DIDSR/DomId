import torch
from domainlab.tasks.utils_task import DsetDomainVecDecorator

from domid.arg_parser import mk_parser_main
from domid.compos.exp.exp_main import Exp
from domid.dsets.make_graph import GraphConstructor
from domid.dsets.make_graph_wsi import GraphConstructorWSI
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.tasks.task_mnist_color import NodeTaskMNISTColor10

# def custom_collate(batch):
#     return {'images': torch.stack([img for img, *_  in batch]),
#             'vec_labels': torch.tensor([vec_y for _, vec_y, *_ in batch]),
#             'vec_d': torch.tensor([[0, 0, 1]for item in batch])}
#
# # Create DataLoader using the custom collate function


def graph_constructor(args):

    graph = GraphConstructor(args.graph_method, 2)

    node = NodeTaskMNISTColor10()
    domain1 = node.get_list_domains()[0]
    dset_tr, dset_val = node.get_dset_by_domain(args, domain1)
    ldr = torch.utils.data.DataLoader(dset_tr)
    bs = args.bs
    X = torch.zeros((len(ldr) * bs, 3, 32, 32))
    label1 = torch.zeros((len(ldr) * bs, 10))
    label2 = torch.zeros((len(ldr) * bs, 10))
    inject_tesnor = torch.zeros((len(ldr) * bs, 0))
    img_id = torch.zeros((len(ldr) * bs, 1))
    start = 0
    for i, (tensor_x, vec_y, *_) in enumerate(ldr):
        end = start + bs
        X[start:end, :, :, :] = tensor_x
        label1[start:end, :] = vec_y
        label2[start:end, :] = vec_y
        start = end

    dataset = torch.utils.data.TensorDataset(X, label1, label2, inject_tesnor, img_id)
    dlr = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
    adjacency_matrices, sparse_matrices = graph.construct_graph(dlr, None)
    return adjacency_matrices, sparse_matrices


def test_MNISTcolor_SDCN_graph_construction_heat():
    print("done")
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--dpath",
            "zout",
            "--task",
            "mnistcolor10",
            "--bs",
            "50",
            "--aname",
            "sdcn",
            "--zd_dim",
            "5",
            "--d_dim",
            "10",
            "--L",
            "5",
            "--prior",
            "Bern",
            "--model",
            "linear",
            "--graph_method",
            "heat",
        ]
    )

    adj_mat, sp_mat = graph_constructor(args)

    for i in adj_mat:
        assert i.shape == (args.bs, args.bs)
    for j in sp_mat:
        assert j.shape == (args.bs, args.bs)


def test_MNISTcolor_SDCN_graph_construction_ncos():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "0",
            "1",
            "2",
            "--dpath",
            "zout",
            "--task",
            "mnist",
            "--bs",
            "50",
            "--aname",
            "sdcn",
            "--zd_dim",
            "5",
            "--d_dim",
            "10",
            "--L",
            "5",
            "--prior",
            "Bern",
            "--model",
            "linear",
            "--graph_method",
            "ncos",
        ]
    )

    adj_mat, sp_mat = graph_constructor(args)

    for i in adj_mat:
        assert i.shape == (args.bs, args.bs)
    for j in sp_mat:
        assert j.shape == (args.bs, args.bs)
