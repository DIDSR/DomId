"""
MNIST
"""

import os

import numpy as np
import torch
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import random


class DsetMNIST(Dataset):
    """
    MNIST Dataset Loading

    - subdomains: MNIST digit value
    - structure: each subdomain contains all images of a given digit
    """

    @store_args
    def __init__(self, digit, args, subset_step=1, list_transforms=None, raw_split="train"):
        """
        :param digit: a integer value from 0 to 9; only images of this digit will be kept.
        :param path: disk storage directory
        :param subset_step: used to subsample the dataset; a fraction of 1/subset_step images is kept
        :param list_transforms: torch transformations
        :param raw_split: default use the training part of mnist
        """
        dpath = os.path.normpath(args.dpath)
        dataset = datasets.MNIST(root=dpath, train=True, download=True, transform=list_transforms)
        # keep only images of specified digit
        self.images = dataset.data[dataset.targets == digit]
        if subset_step == 1 and args.debug:
            # used to speed up the unit tests
            subset_step = 100
        inds_subset = list(range(0, self.images.shape[0], subset_step))
        self.images = self.images[inds_subset]
        n_img = self.images.shape[0]
        # dummy class labels (should not be used; included for consistency with DomainLab)
        self.labels = torch.randint(10, (n_img,), dtype=torch.int32)
        self.args = args

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].numpy()
        image = Image.fromarray(image)
        image = image.convert("RGB")

        # image = Image.open(img_loc)

        if self.list_transforms is not None:
            for trans in self.list_transforms:
                image = trans(image)
        else:
            image = transforms.ToTensor()(image)  # range of pixel [0,1]

        # dummy class labels (should not be used; included for consistency with DomainLab)
        label = self.labels[idx]
        label = mk_fun_label2onehot(10)(label)
        if self.args.dim_inject_y > 0:
            another_label = np.random.randint(0, self.args.dim_inject_y, size=len(label))
        else:
            another_label = np.zeros((len(label)))

        if self.args.path_to_domain:
            inject_domain = np.loadtxt(os.path.join(self.args.path_to_domain, "domain_labels.txt"))[idx]
            # FIXME: no need to hardcode the name of the file as "domain_labels.txt"
            inject_domain = mk_fun_label2onehot(self.args.d_dim)(int(inject_domain) - 1)
            # FIXME: no need to hardcode the number of domains as d_dim
        else:
            inject_domain = np.array([])

        location = ["test"] * len(another_label)

        return (
            image,
            label,
            another_label,
            location,
            inject_domain,
        )  # FIXME for mnist color as well
