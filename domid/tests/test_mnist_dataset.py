from domid.arg_parser import mk_parser_main
from domid.compos.exp.exp_main import Exp
import torch
from domid.models.model_m2yd import ModelXY2D
from domid.models.model_vade import ModelVaDE
from domid.tasks.task_mnist import NodeTaskMNIST
from domid.tasks.task_mnist_color import NodeTaskMNISTColor10

def node_compiler(args):
    if args.task == "mnist":
        node = NodeTaskMNIST()
        domain = "digit2"
    elif args.task == "mnistcolor10":
        node = NodeTaskMNISTColor10()
        domain = "rgb_31_119_180"

    dset2 = node.get_dset_by_domain(args, domain)
    ldr = torch.utils.data.DataLoader(dset2[0]) #train set from the task

    return ldr
def test_mnist_length():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "1"
            "--zd_dim",
            "5",
            "--d_dim",
            "1",
            "--dpath",
            "zout",
            "--L",
            "5",
            "--prior",
            "Bern",
            "--model",
            "linear",
            "--task",
            "mnist"

        ]
    )
    ldr = node_compiler(args)
    it_ldr = iter(ldr)
    x, vec_y, inject_tensor, img_id = next(it_ldr)
    assert x.shape == (1, 3, 32, 32)
    assert vec_y.shape == (1, 10)
    assert inject_tensor == []
    assert len(ldr) == 5958
def test_mnistcolor10_length():
    parser = mk_parser_main()
    args = parser.parse_args(
        [
            "--te_d",
            "7",
            "--tr_d",
            "1",
            "--zd_dim",
            "5",
            "--d_dim",
            "1",
            "--dpath",
            "zout",
            "--L",
            "5",
            "--prior",
            "Bern",
            "--model",
            "linear",
            "--task",
            "mnistcolor10"

        ]
    )

    ldr = node_compiler(args)
    it_ldr = iter(ldr)
    x, vec_y, inject_tensor, img_id = next(it_ldr)
    assert x.shape == (1, 3, 32, 32)
    assert vec_y.shape == (1, 10)
    assert inject_tensor == []
    assert len(ldr) ==600
