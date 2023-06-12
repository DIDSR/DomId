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



class DsetTest(Dataset):


    @store_args
    def __init__(self, digit,args, subset_step=1, list_transforms=None):

        dpath = os.path.normpath(args.dpath)
        self.digit = digit


        self.images = torch.load(os.path.join(dpath, 'images.pt'))
        self.labels = torch.load(os.path.join(dpath, 'labels.pt')).squeeze(1)
        self.images = self.images[self.labels == digit]

        self.args = args
        self.inject_variable = args.inject_var

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #img_loc = os.path.join(self.img_dir, self.images[idx])
        image = self.images[idx]
        # image = Image.fromarray(image)
        # image = image.convert("RGB")

        # if self.list_transforms is not None:
        #     for trans in self.list_transforms:
        #         image = trans(image)
        # else:
        #     image = transforms.ToTensor()(image)




        label = self.labels[idx]

        label = mk_fun_label2onehot(10)(label)
        inject_tensor = []

        img_id = 0

        return image, label, inject_tensor, img_id
