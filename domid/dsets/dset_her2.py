import os

import numpy as np
import pandas as pd
import torch
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DsetHER2(Dataset):
    """
    Dataset of HER2 stained digital microscopy images.
    As currently implemented, the subdomains are the HER2 diagnostic classes 1, 2, and 3.
    There are also 4 data collection site/machine combinations.
    """

    @store_args
    def __init__(self, class_num, path, d_dim, inject_variable=None, metadata=None, transform=None):
        """
        :param class_num: a integer value from 0 to 2, only images of this class will be kept. Note: that actual classes are from 1-3 (therefore, 1 is added in line 28)
        :param path: path to data storage directory (typically passed through args.dpath)
        :param d_dim: number of clusters for the clustering task
        :param inject_variable: name of the variable to be injected for CDVaDE
        :param metadata: path to the CSV file containing the to-be-injected variable for CDVaDE (typecally passed through args.meta_data_csv); if not specified then defaults to "dataframe.csv" in directory given by the "path" argument
        :param transform: torch transformations
        """

        self.dpath = os.path.normpath(path)
        self.list_of_images = []

        self.img_dir = os.path.join(path, "class" + str(class_num + 1) + "jpg")
        self.images = os.listdir(self.img_dir)
        self.class_num = class_num
        self.transform = transform
        self.total_imgs = len(self.images)
        self.inject_variable = inject_variable
        if self.inject_variable is not None:
            if metadata is None:
                self.df = pd.read_csv(os.path.join(path, 'dataframe.csv'))
            else:
                self.df = pd.read_csv(metadata)
            self.u_inject_tensor = len(self.df[self.inject_variable].unique())
        else:
            self.df = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.images[idx])

        image = Image.open(img_loc)
        if self.transform is None:
            self.transform = transforms.ToTensor()
        image = self.transform(image)

        label = mk_fun_label2onehot(3)(self.class_num)

        if self.inject_variable:
            img_info = self.df.loc[self.df['img_id'] == self.images[idx]]
            ind_in_df = img_info.index.item()
            inject_tensor = int(self.df[self.inject_variable][ind_in_df])-1
            inject_tensor = mk_fun_label2onehot(self.u_inject_tensor)(inject_tensor)
        else:
            inject_tensor = [] #torch.Tensor([])#, dtype=label.dtype)

        img_id = self.images[idx]
        return image, label, inject_tensor, img_id
