import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from domid.arg_parser import parse_cmd_args

def run(path):
    data_dir = path
    calculate = True

    if calculate:
        data_transforms = transforms.Compose([transforms.ToTensor()])
        image_datasets = datasets.ImageFolder(os.path.join(data_dir), data_transforms)
        dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=True, num_workers=8)

        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        Mean = mean.div_(len(image_datasets))
        STD = std.div_(len(image_datasets))
        print(Mean)
        print(STD)


def run2(path):
    data_dir = path
    data_transforms = transforms.Compose([transforms.ToTensor()])
    image_datasets = datasets.ImageFolder(os.path.join(data_dir), data_transforms)
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=True, num_workers=8)
    print('==> Computing mean and std..')
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0
    for imgs, targets in dataloader:


        channels_sum += imgs.mean((0, 2, 3))
        channels_squared_sum += (imgs ** 2).mean((0, 2, 3))
        num_batches += 1

    mean = channels_sum / num_batches
    var = (channels_squared_sum / num_batches - mean ** 2)
    std = torch.sqrt(var)
    print(mean)
    print(std)


if __name__ == '__main__':
    args = parse_cmd_args()
    args.dpath
    run2(args.dpath)