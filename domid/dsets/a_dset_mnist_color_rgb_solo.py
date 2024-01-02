"""
Color MNIST with single color
"""
import abc
import os
import struct
from os.path import exists
import torch
import numpy as np
import pandas as pd
from domainlab.dsets.utils_data import mk_fun_label2onehot
from domainlab.utils.utils_class import store_args
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ADsetMNISTColorRGBSolo(Dataset, metaclass=abc.ABCMeta):
    """
    Color MNIST with single color
        1. nominal domains: color palettes/range/spectrum
        2. subdomains: color(foreground, background)
        3. structure: each subdomain contains a combination of foreground+background color
    """

    @abc.abstractmethod
    def get_foreground_color(self, ind):
        raise NotImplementedError

    @abc.abstractmethod
    def get_background_color(self, ind):
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_colors(self):
        raise NotImplementedError

    @store_args
    def __init__(
        self,
        ind_color,
        path,
        subset_step=100,
        color_scheme="both",
        label_transform=mk_fun_label2onehot(10),
        list_transforms=None,
        raw_split="train",
        flag_rand_color=False,
        inject_variable=None,
        args=None,
    ):
        """
        :param ind_color: index of a color palette
        :param path: disk storage directory
        :param color_scheme:
            num (paint according to number),
            back (only paint background),
            both (background and foreground)
        :param list_transforms: torch transformations
        :param raw_split: default use the training part of mnist
        :param flag_rand_color: flag if to randomly paint each image (depreciated)
        :param label_transform:  e.g. index to one hot vector
        """
        dpath = os.path.normpath(path)
        flag_train = True
        if raw_split != "train":
            flag_train = False
        dataset = datasets.MNIST(root=dpath, train=flag_train, download=True, transform=transforms.ToTensor())

        if color_scheme not in ["num", "back", "both"]:
            raise ValueError("color must be either 'num', 'back' or 'both")

        raw_path = os.path.dirname(dataset.raw_folder)
        self._collect_imgs_labels(raw_path, raw_split)
        inds_subset = list(range(0, len(dataset), subset_step))
        self.images = self.images[inds_subset, ::]
        self.labels = self.labels[inds_subset]

        if args.digits_from_mnist:
            self.wanted_digits = [int(w) for w in args.digits_from_mnist]
            self.wanted_digits = list(set(self.wanted_digits))
            self.images, self.labels = self._filter_digits(self.images, self.labels)
        else:
            self.wanted_digits = list(range(10))
        self.label_transform = mk_fun_label2onehot(len(self.wanted_digits))

        self._color_imgs_onehot_labels()

        self.generate_dataframe()
        self.flag_load_df = True
        self.inject_variable = inject_variable
        if self.inject_variable:
            self.inject_dim = args.dim_inject_y

    def _filter_digits(self, x, y=None):
        """
        :param x: a numpy array or a dataframe to be filtered for inclusion of only those digits specified in self.wanted_digits
        :param y: digit labels in the same order as the rows of x; if None then x is assumed to have a 'digit' column.
        :return: filtered x, filtered y
        """
        if y is None:
            assert isinstance(x, pd.DataFrame), "provide a value for y if x is not a pandas dataframe"
            filtered_x = x[x["digit"].isin(self.wanted_digits)]
            filtered_y = filtered_x["digit"].values
        else:
            subidx = []
            for wanted in self.wanted_digits:
                assert 0 <= wanted <= 9
                idx = np.where(y == wanted)[0]
                subidx += list(idx)
            filtered_x = x[subidx]
            filtered_y = y[subidx]
        return filtered_x, filtered_y

    def _collect_imgs_labels(self, path, raw_split):
        """
        :param path:
        :param raw_split:
        """
        if raw_split == "train":
            fimages = os.path.join(path, "raw", "train-images-idx3-ubyte")
            flabels = os.path.join(path, "raw", "train-labels-idx1-ubyte")
        else:
            fimages = os.path.join(path, "raw", "t10k-images-idx3-ubyte")
            flabels = os.path.join(path, "raw", "t10k-labels-idx1-ubyte")

        # Load images
        with open(fimages, "rb") as f_h:
            _, _, rows, cols = struct.unpack(">IIII", f_h.read(16))
            self.images = np.fromfile(f_h, dtype=np.uint8).reshape(-1, rows, cols)

        # Load labels
        with open(flabels, "rb") as f_h:
            struct.unpack(">II", f_h.read(8))
            self.labels = np.fromfile(f_h, dtype=np.int8)
        self.images = np.tile(self.images[:, :, :, np.newaxis], 3)

    def __len__(self):
        return len(self.images)

    def generate_dataframe(self):
        df_name = "dataframe_colored_mnist.csv"
        self.df_save_path = os.path.join(self.path, df_name)
        if exists(self.df_save_path):
            df = pd.read_csv(self.df_save_path)
        else:
            df = pd.DataFrame(columns=["image_id", "color", "digit"])

        for i in range(len(self.images)):
            image_id = "_".join([str(i), str(self.ind_color), str(self.color_scheme), str(self.labels[i])])
            new_row = {"image_id": image_id, "color": self.ind_color, "digit": self.labels[i]}
            # if a row with that image id exists already, replace it; otherwise, add a new row
            indices = df.loc[df["image_id"] == image_id].index
            if len(indices) == 1:
                idx = indices[0]
                for k, v in new_row.items():
                    df.loc[idx, k] = v
            elif len(indices) > 1:
                raise ValueError("Multiple dataframe rows with the same image_id!")
            else:
                df.loc[len(df) + 1] = new_row
        df.to_csv(self.df_save_path, index=False)

    def _op_color_img(self, image):
        """
        transforms raw image into colored version
        """
        # randomcolor is a flag orthogonal to num-back-both
        if self.flag_rand_color:

            c_f = self.get_foreground_color(np.random.randint(0, self.get_num_colors()))
            c_b = 0
            if self.color_scheme == "both":
                count = 0
                while True:
                    c_b = self.get_background_color(np.random.randint(0, self.get_num_colors()))
                    if c_b != c_f and count < 10:
                        # exit loop if background color
                        # is not equal to foreground
                        break
        else:
            if self.color_scheme == "num":
                # domain and class label has perfect mutual information:
                # assign color
                # according to their class (0,10)
                c_f = self.get_foreground_color(self.ind_color)
                c_b = np.array([0] * 3)
            elif self.color_scheme == "back":  # only paint background
                c_f = np.array([0] * 3)
                c_b = self.get_background_color(self.ind_color)

            else:  # paint both background and foreground
                c_f = self.get_foreground_color(self.ind_color)
                c_b = self.get_background_color(self.ind_color)
        image[:, :, 0] = image[:, :, 0] / 255 * c_f[0] + (255 - image[:, :, 0]) / 255 * c_b[0]
        image[:, :, 1] = image[:, :, 1] / 255 * c_f[1] + (255 - image[:, :, 1]) / 255 * c_b[1]
        image[:, :, 2] = image[:, :, 2] / 255 * c_f[2] + (255 - image[:, :, 2]) / 255 * c_b[2]
        return image

    def _color_imgs_onehot_labels(self):
        """
        finish all time consuming operations before torch.dataset.__getitem__
        """
        for i in range(self.images.shape[0]):  # 60k*28*28*3
            img = self.images[i]  # size: 28*28*3 instead of 3*28*28
            self.images[i] = self._op_color_img(img)

    def __getitem__(self, idx):
        if self.flag_load_df:
            self.df = pd.read_csv(self.df_save_path)
            # only keep the single color of this dataset
            self.df = self.df[self.df["color"] == self.ind_color]
            # only keep the digits of this dataset
            if len(self.wanted_digits) < 10:
                self.df, _ = self._filter_digits(self.df)
            self.flag_load_df = False
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        image_id = "_".join([str(idx), str(self.ind_color), str(self.color_scheme), str(self.labels[idx])])
        indices = self.df.loc[self.df["image_id"] == image_id].index

        assert len(indices) == 1, "invalid image_id"
        # this class is for one domain (i.e., one color), but the dataframe combines all domains, so the index will be different:
        df_idx = indices[0]

        image = self.images[idx]  # range of pixel: [0,255]
        label = self.labels[idx]
        if self.label_transform is not None:
            label = self.label_transform(label)
        image = Image.fromarray(image)  # numpy array 28*28*3 -> 3*28*28
        if self.list_transforms is not None:
            for trans in self.list_transforms:
                image = trans(image)
        image = transforms.ToTensor()(image)  # range of pixel [0,1]

        if self.inject_variable:
            inject_tensor = self.df.loc[df_idx, self.inject_variable]
            inject_tensor = mk_fun_label2onehot(self.inject_dim)(inject_tensor)
        else:
            inject_tensor = []

        return image, label, inject_tensor, image_id
