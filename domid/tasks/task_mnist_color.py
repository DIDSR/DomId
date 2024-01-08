"""
Color MNIST with palette
"""

from domainlab.dsets.utils_color_palette import default_rgb_palette  # @FIXME
from domainlab.tasks.utils_task import ImSize
from domainlab.utils.utils_classif import mk_dummy_label_list_str
from torch.utils.data import random_split
from torchvision import transforms

from domid.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domid.tasks.b_task_cluster import NodeTaskDictCluster


class NodeTaskMNISTColor10(NodeTaskDictCluster):
    """
    Use the deafult palette with 10 colors
    """

    def init_business(self, args):
        """
        create a dictionary of datasets
        """
        if args.digits_from_mnist:
            self.wanted_digits = [int(w) for w in args.digits_from_mnist]
            self.wanted_digits = list(set(self.wanted_digits))
        else:
            self.wanted_digits = list(range(10))
        self._dim_y = len(self.wanted_digits)
        # note the order of lines: dim_y has to be set before init_business is called
        super().init_business(args)

    @property
    def dim_y(self):
        return self._dim_y

    @property
    def list_str_y(self):
        return mk_dummy_label_list_str("digit", self.dim_y)

    @property
    def isize(self):
        """image channel, height, width"""
        return ImSize(3, 32, 32)

    def get_list_domains(self):
        """
        1. get list of domain names
        2. better use method than property so new domains can be added
        """
        list_domains = []
        for rgb_list in default_rgb_palette:  # 10 colors
            domain = "_".join([str(c) for c in rgb_list])
            domain = "rgb_" + domain
            list_domains.append(domain)
        return list_domains

    def get_dset_by_domain(self, args, na_domain, split=True):  # @FIXME: different number of arguments than parent
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
        trans = [transforms.Resize((32, 32))]

        dset = DsetMNISTColorSoloDefault(
            ind_global, args.dpath, inject_variable=args.inject_var, list_transforms=trans, args=args
        )

        # split dset into training and test
        if ratio_split>0:
            train_len = int(len(dset) * ratio_split)
            val_len = len(dset) - train_len
            train_set, val_set = random_split(dset, [train_len, val_len])
        else:
            print("no split between train and test datasets")
            train_set = dset
            val_set = dset
        return train_set, val_set
