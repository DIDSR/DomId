import os

import torch
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
class DsetHER2(Dataset):
    """
    Dataset of HER2 stained digital microscopy images.
    As currently implemented, the subdomains are the HER2 diagnostic classes 1, 2, and 3.
    There are also 4 data collection site/machine combinations.
    """

    @store_args
    def __init__(self, class_num, path, d_dim, path_to_domain = None, transform=None):
        """
        :param class_num: a integer value from 0 to 2, only images of this class will be kept.Note: that actual classes are from 1-3 (therefore, 1 is added in line 28)
        :param path: path to root storage directory
        :param d_dim: number of clusters for the clustering task
        :param path_to_domain: if inject previously predicted domain labels, the path needs to be specified.domain_labels.txt must be inside the directory, containing to-be-injected labels.
        :param transform: torch transformations
        """
        self.dpath = os.path.normpath(path)
        self.img_dir = os.path.join(path, "class" + str(class_num + 1) + "jpg")
        self.images = os.listdir(self.img_dir)
        self.class_num = class_num
        self.transform = transform
        self.total_imgs = len(self.images)
        self.path_to_domain = path_to_domain
        self.d_dim = d_dim
        self.loockup_dic = []
        if self.path_to_domain:
            previously_predicted_domain = np.loadtxt(os.path.join(self.path_to_domain, 'domain_labels.txt'))
            previously_predicted_image_path = np.loadtxt(os.path.join(self.path_to_domain, 'image_locs.txt'), str)
            self.lookup_dic = {previously_predicted_image_path[i].split('/')[-1]: int(previously_predicted_domain[i]) for i in range(len(previously_predicted_domain))} #dict{img_path: predicted}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.images[idx])
    
        # import cv2
        # image = cv2.imread(img_loc)
        # image1 = image[:, :, ::-1]
        # im = torch.flip(torch.from_numpy(image.copy()), dims=(2,))

        image = Image.open(img_loc)
        if self.transform is None:
            self.transform = transforms.ToTensor()
        image = self.transform(image)

        # mean = [image[0, :, :].mean(), image[1, :, :].mean(), image[2, :, :].mean()]
        #
        # m = torch.zeros(image.shape[0], image.shape[1], image.shape[2])
        # m[0, :, :] = mean[0]
        # m[1, :, :] = mean[1]
        # m[2, :, :] = mean[2]
        #
        # norm_img = image - m
        # image = norm_img

        # return the one-hot encoded label

        label = mk_fun_label2onehot(3)(self.class_num) #FIXME 3
        #A_FDA, A_NIH, H1, H2
        machine = img_loc[-6:-4]

        if self.path_to_domain:
            #domain = np.loadtxt(os.path.join(self.path_to_domain, 'domain_labels.txt'))[idx]
            # FIXME: no need to hardcode the name of the file as "domain_labels.txt"
   
            domain = self.lookup_dic[self.images[idx]]
            domain = mk_fun_label2onehot(self.d_dim)(int(domain)-1)

            # FIXME: no need to hardcode the number of domains as d_dim
        else:
            domain = []
        return image, label, machine, img_loc, domain
