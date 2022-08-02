"""
HER2 task where the HER2 categories are considered "domains"
"""

from torch.utils.data import random_split
from torchvision import transforms
from libDG.libdg.tasks.utils_task import ImSize
from libDG.libdg.utils.utils_classif import mk_dummy_label_list_str
#from libDG.libdg.tasks.b_task import NodeTaskDict
from domid.tasks.b_task import NodeTaskDict
from domid.dsets.dset_her2 import DsetHER2


class NodeTaskHER2(NodeTaskDict):
    """
    Based on NodeTaskMNISTColor10 from libDG.
    The digits (0, 1, ..., 9) are regarded as domains (to be separated by unsupervised clustering).
    """

    # def init_business(self, a):
    #     print('i do not know what this function is')
    @property
    def list_str_y(self):
        """
        MNIST task has no labels (digits are considered domains)
        """
        #['FD', 'ND', 'H1', 'H2']
        return mk_dummy_label_list_str("dummy", 3)

    @property
    def isize(self):
        """image channel, height, width"""
        return ImSize(3, 100, 100)  # FIXME should be in sync with transforms

    def get_list_domains(self):
        """
        Get list of domain names
        """
        return mk_dummy_label_list_str("class", 3)

    def get_dset_by_domain(self, args, na_domain, split=True):  # , na_domain, split=True):
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
        ind_global = self.get_list_domains().index(na_domain)
        mean = [0.6399, 0.5951, 0.6179]
        std = [0.1800, 0.1980, 0.2070] #[0.1582, 0.1728, 0.1728]
        #mean = [0.4, 0.4, 0.4]
        #mean = [0.5, 0.5, 0.5]
        #confirm mean 0 and std 1 after the normalization
        trans = transforms.Compose([transforms.Resize((100, 100)), transforms.RandomHorizontalFlip(),transforms.ToTensor()])#, transforms.Normalize(mean, std)])
        dset = DsetHER2(ind_global, args.dpath, transform=trans)

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
    args = parser.parse_args(["--te_d", "0", "--dpath", "./HER2/combined_train", "--split", "0.2"])
    print(args)
    node = NodeTaskHER2()
    na_domain = 3  # ['0', '1', '2']
    node.get_dset_by_domain(args)  # , na_domain)
    print(node.get_list_domains())
    node.list_str_y
    node.init_business(args)
