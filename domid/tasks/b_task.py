"""
Use dictionaries to create train and test domain split
"""
from collections import Counter
import torch
from torch.utils.data.dataset import ConcatDataset

from libdg.tasks.a_task import NodeTaskDGClassif
from libdg.tasks.utils_task import mk_loader, mk_onehot, DsetDomainVecDecorator, DsetDomainVecDecoratorImgPath
from libdg.tasks.utils_task_dset import DsetIndDecorator4XYD


def dset_decoration_args_algo(args, ddset):
    if "match" in args.aname:  # FIXME: are there ways not to use this if statement?
            ddset = DsetIndDecorator4XYD(ddset)
    return ddset


class NodeTaskDict(NodeTaskDGClassif):
    """
    Use dictionaries to create train and test domain split
    """
    @property
    def list_str_y(self):
        return self._list_str_y

    @list_str_y.setter
    def list_str_y(self, list_str_y):
        self._list_str_y = list_str_y

    @property
    def isize(self):
        return self._im_size

    @isize.setter
    def isize(self, im_size):
        self._im_size = im_size

    def get_list_domains(self):
        return self._list_domains

    def set_list_domains(self, list_domains):
        self._list_domains = list_domains

    def get_dset_by_domain(self, args, na_domain):
        raise NotImplementedError

    def init_business(self, args):
        """
        create a dictionary of datasets
        """
        list_domain_tr, list_domain_te = self.get_list_domains_tr_te(args.tr_d, args.te_d)
        self.dict_dset = dict()
        self.dict_dset_val = dict()
        dim_d = len(list_domain_tr)
        D_tr =[]
        D_val = []
        for (ind_domain_dummy, na_domain) in enumerate(list_domain_tr):
            dset_tr, dset_val = self.get_dset_by_domain(args, na_domain)
            D_tr+=dset_tr
            D_val += dset_val
        ddset_mix = D_tr

        self._loader_tr = mk_loader(ddset_mix, args.bs)
        ddset_mix_val = D_val
        self._loader_val = mk_loader(ddset_mix_val, args.bs)
        self._loader_te = mk_loader(ddset_mix_val, args.bs) #FIXME
        #self.count_domain_class()

    def count_domain_class(self):
        """
        iterate all domains and count the class label distribution for each
        return a double dictionary {"domain1": {"class1":3, "class2": 4,...}, ....}
        """
        for key, dset in self.dict_dset.items():
            dict_class_count = self._count_class_one_hot(dset)
            self.dict_domain_class_count[key] = dict_class_count
        for key, dset in self.dict_dset_te.items():
            dict_class_count = self._count_class_one_hot(dset)
            self.dict_domain_class_count[key] = dict_class_count

    def _count_class_one_hot(self, dset):
        labels_count = torch.zeros(self.dim_y, dtype=torch.long)
        for _, target, *_ in dset:
            labels_count += target.long()

        list_count = list(labels_count.cpu().numpy())
        dict_class_count = dict()
        for name, count in zip(self.list_str_y, list_count):
            dict_class_count[name] = count
        return dict_class_count

    def _count_class(self, dset):   # FIXME: remove this
        labels = dset.targets
        class_dict = dict(Counter(labels))
        return class_dict
