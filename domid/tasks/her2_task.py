"""
Basic MNIST task where the digits are considered "domains"
"""

from torch.utils.data import random_split
from libDG.libdg.tasks.utils_task import DsetDomainVecDecorator, mk_onehot, mk_loader, ImSize
from libDG.libdg.utils.utils_classif import mk_dummy_label_list_str
from libDG.libdg.tasks.task_mnist_color import NodeTaskMNISTColor10
from domid.dsets.dset_her2 import DsetHER2


class NodeTaskHER2():
    """
    Based on NodeTaskMNISTColor10 from libDG.
    The digits (0, 1, ..., 9) are regarded as domains (to be separated by unsupervised clustering).
    """
    def init_business(self, a):
        print('i do not know what this function is')
    @property
    def list_str_y(self):
        """
        MNIST task has no labels (digits are considered domains)
        """
        return mk_dummy_label_list_str("dummy", 3)

    @property
    def isize(self):
        """image channel, height, width"""
        return ImSize(3, 28, 28)

    def get_list_domains(self):
        """
        Get list of domain names
        """
        return mk_dummy_label_list_str("digit", 3)

    def get_dset_by_domain(self, args): #, na_domain, split=True):
        """get_dset_by_domain.
        :param args:
        :param na_domain:
        :param split: for test set, no need to split
        args.split: by default, split is set to be zero which in python can
        be evaluated in if statement, in which case, no validation set will be
        created. Otherwise, this argument is the split ratio
        """
        split = True
        ratio_split = float(args.split) if split else False
        # by default, split is set to be zero which in python can
        # be evaluated in if statement, in which case, no validation
        # set will be created. Otherwise, this argument is
        # the split ratio
        ind_global = self.get_list_domains() #.index(na_domain)
        print('IND global', ind_global)
        dset = DsetHER2(args.dpath, 1 , None)
        train_set = dset
        val_set = dset
        # split dset into training and validation sets
        if ratio_split:
            train_len = int(len(dset) * ratio_split)
            val_len = len(dset) - train_len
            train_set, val_set = random_split(dset, [train_len, val_len])
        return train_set, val_set

def test_fun():
    from libdg.arg_parser import mk_parser_main

    parser = mk_parser_main()
    print(parser)
    args = parser.parse_args(["--te_d", "0", "--dpath", "./HER2/Testing_fixed/categorized/combined_train/*jpg", "--split", "0.2"])
    print(args)
    node = NodeTaskHER2()
    na_domain = 3 #['0', '1', '2']
    node.get_dset_by_domain(args) #, na_domain)
    print(node.get_list_domains())
    node.list_str_y
    node.init_business(args)

