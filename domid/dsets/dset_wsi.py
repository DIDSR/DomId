import os

import numpy as np
import pandas as pd
import torch
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class DsetWSI(Dataset):
    """
    Dataset of WEAH stained digital microscopy images.
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
        self.df = pd.read_csv(args.meta_data_csv)
        print("the data is loading from the csv:", args.meta_data_csv)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print(self.images[idx])
        # print(self.images[idx])
        # import pdb; pdb.set_trace()
        img_loc = os.path.join(self.dpath, self.images[idx])

        # print(img_loc)
        image = Image.open(img_loc)
        # print(image)
        if self.transform:
            for trans in self.transform:
                image = trans(image)

        image = transforms.ToTensor()(image)

        resp_label = int(self.df.loc[self.df["path"] == self.images[idx]]["resp"])
        cah_label = int(self.df.loc[self.df["path"] == self.images[idx]]["ann"])

        label_dict = {"01": 0, "02": 1, "11": 2, "12": 3, "03": 4, "13": 5}
        encod_label = label_dict[str(resp_label) + str(cah_label)]

        label = mk_fun_label2onehot(6)(encod_label)
        inject_tensor = []
        img_id = img_loc

        return image, label, inject_tensor, img_id
