import torch
from domid.arg_parser import mk_parser_main
from domid.compos.exp.exp_main import Exp
from domid.dsets.make_graph import GraphConstructor
from domid.dsets.make_graph_wsi import GraphConstructorWSI
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.tasks.task_mnist_color import NodeTaskMNISTColor10
from domainlab.tasks.utils_task import DsetDomainVecDecorator
def custom_collate(batch):
    return {'images': torch.stack([img for img, *_  in batch]),
            'vec_labels': torch.tensor([vec_y for _, vec_y, *_ in batch]),
            'vec_d': torch.tensor([[0, 0, 1]for item in batch])}

# Create DataLoader using the custom collate function


def graph_constructor(args):

    graph = GraphConstructor("heat", 7)

    node = NodeTaskMNISTColor10()
    domain1 = node.get_list_domains()[0]
    dset1 = node.get_dset_by_domain(args, domain1)
    ldr = torch.utils.data.DataLoader(dset1[0])
    graph.construct_graph(ldr, None)
    return graph


def test_MNISTcolor_SDCN_graph_construction():
    print('done')
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

    graph_constructor(args)

def test_MNISTcolor_SDCN_graph_construction():
    print('done')
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
            "heat",

        ]
    )

    graph_constructor(args)