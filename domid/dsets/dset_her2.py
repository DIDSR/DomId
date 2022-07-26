import os

from PIL import Image

from torch.utils.data import Dataset
from libdg.utils.utils_class import store_args
from libdg.dsets.utils_data import mk_fun_label2onehot


class DsetHER2(Dataset):
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
        image = self.transform(image)
        label = mk_fun_label2onehot(3)(self.class_num)

        return image, label
