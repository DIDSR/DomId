import os

import torch
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
class DsetHER2(Dataset):
    """
    Dataset of HER2 stained digital microscopy images.
    As currently implemented, the subdomains are the HER2 diagnostic classes 1, 2, and 3.
    There are also 4 data collection site/machine combinations.
    """

    @store_args
    def __init__(self, class_num, path, d_dim, inject_variabe = None, transform=None):
        """
        :param class_num: a integer value from 0 to 2, only images of this class will be kept.Note: that actual classes are from 1-3 (therefore, 1 is added in line 28)
        :param path: path to root storage directory
        :param d_dim: number of clusters for the clustering task
        :param path_to_domain: if inject previously predicted domain labels, the path needs to be specified.domain_labels.txt must be inside the directory, containing to-be-injected labels.
        :param transform: torch transformations
        """

        self.dpath = os.path.normpath(path)
        self.list_of_images = []
        # for folder in os.listdir(self.dpath):
        #
        #     folder_path = os.path.join(path, folder)
        #     if os.path.isdir(folder_path):
        #         self.list_of_images += [os.path.join(path, folder, image) for image in os.listdir(folder_path)]

        self.img_dir = os.path.join(path, "class" + str(class_num + 1) + "jpg")
        self.images = os.listdir(self.img_dir)
        self.class_num = class_num
        self.transform = transform
        self.total_imgs = len(self.images)
        #self.path_to_domain = path_to_domain
        # self.d_dim = d_dim
        #self.loockup_dic = []
        self.df = pd.read_csv(os.path.join(path, 'dataframe.csv'))
        self.inject_variable = inject_variabe




        #
        # if self.path_to_domain:
        #     previously_predicted_domain = np.loadtxt(os.path.join(self.path_to_domain, 'domain_labels.txt'))
        #     previously_predicted_image_path = np.loadtxt(os.path.join(self.path_to_domain, 'image_locs.txt'), str)
        #     self.lookup_dic = {previously_predicted_image_path[i].split('/')[-1]: int(previously_predicted_domain[i]) for i in range(len(previously_predicted_domain))} #dict{img_path: predicted}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_loc = os.path.join(self.img_dir, self.images[idx])

        image = Image.open(img_loc)
        if self.transform is None:
            self.transform = transforms.ToTensor()
        image = self.transform(image)


        img_info = self.df.loc[self.df['img_id'] == self.images[idx]]
        ind_in_df =img_info.index.item()
        label = mk_fun_label2onehot(3)(self.class_num)

        if self.inject_variable:
            inject_tensor = int(self.df[self.inject_variable][ind_in_df][-4])-1

            u_inject_tensor = len(self.df[self.inject_variable].unique())
            inject_tensor = mk_fun_label2onehot(u_inject_tensor)(inject_tensor)
        else:
            inject_tensor = [] #torch.Tensor([])#, dtype=label.dtype)





        img_id = self.images[idx]
        return image, label, inject_tensor, img_id
