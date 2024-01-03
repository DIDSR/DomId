# from domainlab.tasks.task_mnist_color import NodeTaskMNISTColor10
import os

import numpy as np
import pandas as pd
from domainlab.tasks.b_task import NodeTaskDict
from domainlab.tasks.utils_task import DsetDomainVecDecorator, ImSize, mk_loader, mk_onehot
from domainlab.utils.utils_classif import mk_dummy_label_list_str
from torch.utils.data import random_split
from torchvision import transforms

from domid.dsets.dset_wsi import DsetWSI
from domid.tasks.b_task_cluster import NodeTaskDictCluster


class NodeTaskWSI(NodeTaskDictCluster):
    @property
    def list_str_y(self):
        """
        WEAH task has no labels (digits are considered domains)
        """
        return mk_dummy_label_list_str("dummy", 6)

    @property
    def isize(self):
        """
        :return: image size object storing image channels, height, width.
        """
        return ImSize(3, 64, 64)

    def get_list_domains(self):
        """
        Get list of domain names

        :return: list of domain names
        """
        return mk_dummy_label_list_str("digit", 313)

    def get_dset_by_domain(self, args, na_domain, split=True):
        """Get a dataset by digit

        :param args: command line arguments
        :param na_domain: domain name
        :param split: whether a training/validation split is performed (the
        training split portion will be determined by args.split); for test
        set, no need to split; args.split: by default, split is set to be
        zero which in python can be evaluated in if statement, in which case,
        no separate validation set will be created. Otherwise, this argument
        is the percentage of the data to be used as training set, while the
        rest will be used as validation set.
        :return: training dataset, validation dataset
        """
        ratio_split = float(args.split) if split else False
        # by default, split is set to be zero which in python can
        # be evaluated in if statement, in which case, no validation
        # set will be created. Otherwise, this argument is
        # the split ratio
        dpath = args.dpath  # png_files/32, 16))]  # , transforms.ToTensor()]
        ind_global = self.get_list_domains().index(na_domain)
        df = pd.read_csv(args.meta_data_csv)
        mask = df[df["subject"] == ind_global]
        # mask = df['resp'].values == ind_global  # response (0 o 1)
        img_paths = np.array(mask["path"])  # [:400]

        trans = [transforms.Resize((64, 64))]
        dset = DsetWSI(class_num=ind_global, path=img_paths, args=args, transform=trans)
        train_set = dset
        val_set = dset
        # split dset into training and validation sets
        # if ratio_split:
        #     train_len = int(len(dset) * ratio_split)
        #     val_len = len(dset) - train_len
        #     train_set, val_set = random_split(dset, [train_len, val_len])
        return train_set, val_set


def test_fun():
    from domainlab.arg_parser import mk_parser_main

    parser = mk_parser_main()
    args = parser.parse_args(["--te_d", "0", "--dpath", "zout", "--split", "0.2"])
    node = NodeTaskMNIST()
    node.get_list_domains()
    node.list_str_y
    node.init_business(args)
