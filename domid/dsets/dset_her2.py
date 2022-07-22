"""
MNIST
"""

import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from libdg.dsets.utils_data import mk_fun_label2onehot
from libdg.utils.utils_class import store_args


class DsetHER2(Dataset):

    @store_args
    def __init__(self, path, subset_step = 1, transform = None):

        dpath = os.path.normpath(path)
        print('Initialization')
        self.images = datasets.ImageFolder(dpath)
        print('after self.images')
        #breakpoint()

        # #dataset = datasets.MNIST(root=dpath,
        #                          train=True,
        #                          download=True,
        #                          transform=transforms.ToTensor())
        # keep only images of specified digit
        #self.images = dataset
        #inds_subset = list(range(0, self.images.shape[0], subset_step))
        #self.images = self.images[inds_subset]
        #n_img = self.images.shape[0]
        # dummy class labels (should not be used; included for consistency with libDG)
        #self.labels = torch.randint(10, (n_img,), dtype=torch.int32)
        import pandas as pd
        self.img_labels = self.images.class_to_idx# pd.read_csv(annotations_file)
        #path = "./HER2/Testing_fixed/categorized/combined_train/*jpg"
        self.img_dir = path
        self.transform = transforms.ToTensor()

        #dataset = None
        #self.target_transform = target_transform



    def __len__(self):
        return len(self.images)
    #

    #
    #
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels)
        print('here 2')
        image = read_image(img_path)
        print(image)
        #label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    # def __getitem__(self, idx):
    #     image = self.images[idx].numpy()
    #     image = Image.fromarray(image)
    #     image = image.convert('RGB')
    #     if self.list_transforms is not None:
    #         for trans in self.list_transforms:
    #             image = trans(image)
    #     image = transforms.ToTensor()(image)  # range of pixel [0,1]
    #
    #     # dummy class labels (should not be used; included for consistency with libDG)
    #     label = self.labels[idx]
    #     label = mk_fun_label2onehot(10)(label)
    #     return image, label
