"""
Basic MNIST task where the digits are considered "domains"
"""

from torch.utils.data import random_split
from DomainLab.domainlab.tasks.utils_task import DsetDomainVecDecorator, mk_onehot, mk_loader, ImSize
from DomainLab.domainlab.utils.utils_classif import mk_dummy_label_list_str
from DomainLab.domainlab.tasks.task_mnist_color import NodeTaskMNISTColor10
from domid.dsets.dset_mnist import DsetMNIST


class NodeTaskMNIST(NodeTaskMNISTColor10):
    """
    Based on NodeTaskMNISTColor10 from DomainLab.
    The digits (0, 1, ..., 9) are regarded as domains (to be separated by unsupervised clustering).
    """
    @property
    def list_str_y(self):
        """
        MNIST task has no labels (digits are considered domains)
        """
        return mk_dummy_label_list_str("dummy", 10)

    @property
    def isize(self):
        """image channel, height, width"""
        return ImSize(3, 28, 28)

    def get_list_domains(self):
        """
        Get list of domain names
        """
        return mk_dummy_label_list_str("digit", 10)

    def get_dset_by_domain(self, args, na_domain, split=True):
        """get_dset_by_domain.
        :param args:
        :param na_domain:
        :param split: for test set, no need to split
        args.split: by default, split is set to be zero which in python can
        be evaluated in if statement, in which case, no validation set will be
        created. Otherwise, this argument is the split ratio
        """
        ratio_split = float(args.split) if split else False
        # by default, split is set to be zero which in python can
        # be evaluated in if statement, in which case, no validation
        # set will be created. Otherwise, this argument is
        # the split ratio
        ind_global = self.get_list_domains().index(na_domain)
        dset = DsetMNIST(ind_global, args.dpath)
        train_set = dset
        val_set = dset
        # split dset into training and validation sets
        if ratio_split:
            train_len = int(len(dset) * ratio_split)
            val_len = len(dset) - train_len
            train_set, val_set = random_split(dset, [train_len, val_len])
        return train_set, val_set

def test_fun():
    from domainlab.arg_parser import mk_parser_main
    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "0", "--dpath", "zout", "--split", "0.2"])
    node = NodeTaskMNIST()
    node.get_list_domains()
    node.list_str_y
    node.init_business(args)
