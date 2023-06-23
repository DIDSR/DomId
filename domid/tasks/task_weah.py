# from domainlab.tasks.task_mnist_color import NodeTaskMNISTColor10
from domainlab.tasks.b_task import NodeTaskDict
from domainlab.tasks.utils_task import (DsetDomainVecDecorator, ImSize,
                                        mk_loader, mk_onehot)
from domainlab.utils.utils_classif import mk_dummy_label_list_str
from torch.utils.data import random_split
import os
from domid.dsets.dset_weah import DsetWEAH
from torchvision import transforms
import pandas as pd
import numpy as np


class NodeTaskWEAH(NodeTaskDict):


    @property
    def list_str_y(self):
        """
        WEAH task has no labels (digits are considered domains)
        """
        return mk_dummy_label_list_str("dummy", 4)

    @property
    def isize(self):
        """
        :return: image size object storing image channels, height, width.
        """
        return ImSize(3, 256, 256)

    def get_list_domains(self):
        """
        Get list of domain names

        :return: list of domain names
        """
        return mk_dummy_label_list_str("digit", 4)

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
        dpath = args.dpath  # png_files/Training

        #         path1 = []
        #         path2 = []
        #         path3 = []
        #         dpath = args.dpath
        #         for folder in os.listdir(dpath):
        #             folder_path = os.path.join(dpath, folder)
        #             for file in os.listdir(folder_path):

        #                 if file[-3:] == 'png':
        #                     annotation = file.split('_')[2]
        #                     if annotation == '1':
        #                         path1.append(os.path.join(dpath, folder, file))
        #                     if annotation == '2':
        #                         path2.append(os.path.join(dpath, folder, file))
        #                     if annotation == '3':
        #                         path3.append(os.path.join(dpath, folder, file))

        #         #print(len(path1), len(path2), len(path3))
        #         paths = [path1[:500], path2[:500], path3[:500]]
        trans = [transforms.Resize((256, 256))]  # , transforms.ToTensor()]
        ind_global = self.get_list_domains().index(na_domain)
        df = pd.read_csv('../dset_WEAH.csv')
        mask = df['resp'].values == ind_global  # response (0 o 1)
        img_paths = np.array(df.loc[mask]['path'])  # [:400]

        dset = DsetWEAH(class_num=ind_global, path=img_paths, args=args, transform=trans)
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
