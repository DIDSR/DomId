import os

import torch
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
import pandas as pd


class DsetWEAH(Dataset):
    """
    Dataset of HER2 stained digital microscopy images.
    As currently implemented, the subdomains are the HER2 diagnostic classes 1, 2, and 3.
    There are also 4 data collection site/machine combinations.
    """

    @store_args
    def __init__(self, class_num, path, args, path_to_domain=None, transform=None):
        """
        :param class_num: a integer value from 0 to 2, only images of this class will be kept.Note: that actual classes are from 1-3 (therefore, 1 is added in line 28)
        :param path: path to root storage directory
        :param d_dim: number of clusters for the clustering task
        :param path_to_domain: if inject previously predicted domain labels, the path needs to be specified.domain_labels.txt must be inside the directory, containing to-be-injected labels.
        :param transform: torch transformations
        """
        self.dpath = args.dpath
        self.img_dir = args.dpath  # os.path.join(path, "class" + str(class_num + 1) + "jpg")
        self.images = path  # os.listdir(self.img_dir)
        self.class_num = class_num
        self.transform = transform
        self.total_imgs = len(self.images)
        self.path_to_domain = path_to_domain
        self.d_dim = args.d_dim
        self.df = pd.read_csv('../dset_WEAH.csv')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print(self.images[idx])
        # print(self.images[idx])
        img_loc = os.path.join(self.dpath, self.images[idx][:12], self.images[idx])

        # print(img_loc)
        image = Image.open(img_loc)
        # print(image)
        if self.transform:
            for trans in self.transform:
                image = trans(image)

        image = transforms.ToTensor()(image)
        # label = mk_fun_label2onehot(3)(self.class_num) #FIXME: responded and non responded
        # A_FDA, A_NIH, H1, H2

        # print(p.read_csv('../dset_WEAH.csv'))
        resp_label = int(self.df.loc[self.df['path'] == self.images[idx]]['resp'])
        cah_label = int(self.df.loc[self.df['path'] == self.images[idx]]['CAH'])
        label = torch.cat((mk_fun_label2onehot(2)(resp_label), mk_fun_label2onehot(2)(cah_label)), 0)

        BMI = self.df.loc[(self.df['path'] == self.images[idx])]['BMI']  # BMIinstead of machine
        BMI = int(BMI)

        if self.path_to_domain:
            domain = np.loadtxt(os.path.join(self.path_to_domain, 'domain_labels.txt'))[idx]
            # FIXME: no need to hardcode the name of the file as "domain_labels.txt"
            domain = mk_fun_label2onehot(self.d_dim)(int(domain) - 1)
            # FIXME: no need to hardcode the number of domains as d_dim
        else:
            domain = []
        # print('label', label)

        return image, label, BMI, img_loc, domain