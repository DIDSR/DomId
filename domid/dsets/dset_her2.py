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
    def __init__(self, class_num, path, subset_step = 1, transform = None):

        self.dpath = os.path.normpath(path)
        print('Initialization')
        #breakpoint()
        dataset = datasets.ImageFolder(self.dpath)#, transform = transforms.ToTensor()) #672 samples
        print('after self.images')
        #breakpoint()

        # #dataset = datasets.MNIST(root=dpath,
        #                          train=True,
        #                          download=True,
        #                          transform=transforms.ToTensor())
        # keep only images of specified digit
        #breakpoint()
        self.images = dataset[dataset.targets==class_num] #FIXME dataset.data
        _, counts = torch.unique(torch.Tensor(dataset.targets), return_counts=True)
        n_img = counts[class_num]
        #n_img = 2 #672 #torch.unique(targets, return_counts=True)

        # dummy class labels (should not be used; included for consistency with libDG)
        self.images = dataset
        self.img_labels = torch.randint(3, (n_img,), dtype=torch.int32)
        import pandas as pd
        #self.img_labels = self.images.class_to_idx# pd.read_csv(annotations_file)
        #path = "./HER2/Testing_fixed/categorized/combined_train/*jpg"
        self.img_dir = os.path.join(path, 'class'+str(class_num+1)+'jpg')

        self.transform = None

        #dataset = None
        #self.target_transform = target_transform



    def __len__(self):
        return len(self.images)
    #

    #
    #
    def __getitem__(self, idx):
        #breakpoint()
        #print(idx)
        # image = self.images[idx]
        # import torch
        # image = torch.Tensor(image)
        #breakpoint()
        idx = 0
        img_name= os.listdir(self.img_dir)[idx]
        img_path = os.path.join(self.img_dir, img_name)
        #img_path = './HER2/combined_train/class1jpg/6670-6027,6543FD.jpg'
        #img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = read_image(img_path)/1000 #FIXME needs transform
        label = self.img_labels[idx]
        #image = np.asarray(image)
        # image = Image.fromarray(image)
        # image = image.convert('RGB')
        # if self.list_transforms is not None:
        #     for trans in self.list_transforms:
        #         image = trans(image)
        #image = transforms.ToTensor()(image)  # range of pixel [0,1]

        # dummy class labels (should not be used; included for consistency with libDG)
        label = torch.randint(0, 1, (1,)) #self.img_labels[idx]
        #label = mk_fun_label2onehot(10)(label)
        #breakpoint()



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
