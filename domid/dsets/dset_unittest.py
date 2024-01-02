import os

import numpy as np
import pandas as pd
import torch
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import shutil


class DsetUnitTest(Dataset):
    """
    This dataset is solely used for unit testing of loss values.
    The images contain tensors of one with the dimension of 1x16x16, the label is a random integer.
    """

    @store_args
    def __init__(self, digit, args, subset_step=1, list_transforms=None):

        dpath = os.path.normpath(args.dpath)
        self.digit = digit

        if not os.path.exists(dpath):
            self.create_the_dataset(dpath)

        self.images = torch.load(os.path.join(dpath, "images.pt"))
        self.labels = torch.load(os.path.join(dpath, "labels.pt")).squeeze(1)
        self.images = self.images[self.labels == digit]

        self.args = args
        self.inject_variable = args.inject_var

    def create_the_dataset(self, dpath):
        # Check if the directory exists
        if not os.path.exists(dpath):
            os.makedirs(dpath)
            dummy_images = torch.ones(7000, 3, 16, 16)
            dummy_labels = torch.randint(0, 10, (7000, 1))
            torch.save(dummy_images, os.path.join(dpath, "images.pt"))
            torch.save(dummy_labels, os.path.join(dpath, "labels.pt"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        label = mk_fun_label2onehot(10)(label)
        inject_tensor = []

        img_id = 0

        return image, label, inject_tensor, img_id
