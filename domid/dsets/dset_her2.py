import os

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
    def __init__(self, class_num, path, transform=None):
        self.dpath = os.path.normpath(path)
        self.img_dir = os.path.join(path, "class" + str(class_num + 1) + "jpg")
        self.images = os.listdir(self.img_dir)
        self.class_num = class_num
        self.transform = transform
        self.total_imgs = len(self.images)

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

        return image, label, machine, img_loc
