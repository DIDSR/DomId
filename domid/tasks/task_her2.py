from domainlab.tasks.utils_task import ImSize
from domainlab.utils.utils_classif import mk_dummy_label_list_str
from torch.utils.data import random_split
from torchvision import transforms

from domid.tasks.b_task_cluster import NodeTaskDictCluster
from domid.dsets.dset_her2 import DsetHER2
from domid.utils.perf_similarity import PerfCorrelationHER2


class NodeTaskHER2(NodeTaskDictCluster):
    """
    HER2 task where the HER2 categories are considered "domains"

    """

    @property
    def list_str_y(self):
        """
        Labels are not used in clustering. So, we just return a dummy list for now (for compatibility with domainlab).
        """
        return mk_dummy_label_list_str("dummy", 3)

    @property
    def isize(self):
        """
        :return: image size object storing image channels, height, width.
        """
        return ImSize(3, 32, 32)  # FIXME should be in sync with transforms

    def get_list_domains(self):
        """
        Get list of domain names
        :return: list of domain names
        """
        return mk_dummy_label_list_str("class", 3)

    def get_dset_by_domain(self, args, na_domain, split=True):
        """Get a dataset by domain name
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
        ind_global = self.get_list_domains().index(na_domain)
        # Calculated std amd mean values are computed using the code
        # in utils/mean_std.py. Those are the average mean and std values
        # for HER2 training images by channel.
        # mean = [0.6399, 0.5951, 0.6179]
        # std = [0.1800, 0.1980, 0.2070]

        trans = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAutocontrast(0.25),
                transforms.RandomAdjustSharpness(2, 0.25),
                transforms.ToTensor(),
            ]
        )

        dset = DsetHER2(ind_global, args.dpath, args.d_dim, args.inject_var, transform=trans)
        train_set = dset
        val_set = dset
        # split dset into training and validation sets
        if ratio_split:
            train_len = int(len(dset) * ratio_split)
            val_len = len(dset) - train_len
            train_set, val_set = random_split(dset, [train_len, val_len])

        return train_set, val_set

    def calc_corr(self, model, loader_tr, loader_te, device):
        perf_metric_correlation = PerfCorrelationHER2()
        r_score_tr = perf_metric_correlation.cal_acc(model, loader_tr, device)
        # cal_acc(clc, model, loader_tr, device, max_batches=None):
        r_score_te = perf_metric_correlation.cal_acc(model, loader_te, device)
        return r_score_tr, r_score_te


def test_fun():
    from domainlab.arg_parser import mk_parser_main

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
