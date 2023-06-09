import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

# ex = '2022-10-20 12:09:05.420792/'
ex = "2022-10-20 15:09:14.352312/"

writer = SummaryWriter("tsne_tuning/" + ex)

prediction = np.loadtxt(ex + "domain_labels.txt")[:-1]
true = np.loadtxt(ex + "machine_labels.txt", dtype=str)
Z = np.load(ex + "Z_space.npy")[:-1, :]
# prediction = true
img = torch.rand(len(prediction), 3, 10, 10)

for i in range(len(prediction)):
    if prediction[i] == "H1":
        img[i, 0, :, :] = torch.ones(10, 10)

    if prediction[i] == "H2":
        img[i, 1, :, :] = torch.ones(10, 10)

    if prediction[i] == "FD":
        img[i, 2, :, :] = torch.ones(10, 10)

    if prediction[i] == "ND":

        img[i, 1, :, :] = torch.ones(10, 10) / 4
        img[i, 2, :, :] = torch.ones(10, 10) / 4
img = torch.Tensor(img)
print(img)
writer.add_embedding(Z, label_img=img, metadata=prediction)  # , tag = ex[:-1])
