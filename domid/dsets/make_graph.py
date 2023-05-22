import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
topk = 10

def construct_graph(features, label, method='heat'):
    fname = '../graph/usps_custom_graph.txt'

    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))

'''
f = h5py.File('data/usps.h5', 'r')
train = f.get('train')
test = f.get('test')
X_tr = train.get('data')[:]
y_tr = train.get('target')[:]
X_te = test.get('data')[:]
y_te = test.get('target')[:]
f.close()
usps = np.concatenate((X_tr, X_te)).astype(np.float32)
label = np.concatenate((y_tr, y_te)).astype(np.int32)
'''

'''
hhar = np.loadtxt('data/hhar.txt', dtype=float)
label = np.loadtxt('data/hhar_label.txt', dtype=int)
'''
#
# reut = np.loadtxt('data/reut.txt', dtype=float)
# label = np.loadtxt('data/reut_label.txt', dtype=int)
dpath = './zdoutput/'
dataset = datasets.USPS(root=dpath, train=True, download=True, transform=None)
features = dataset.data
features = torch.tensor(features, dtype=torch.float32)
rgb_tensor= torch.stack([features] * 3, dim=1)
flattened_features = rgb_tensor.reshape(rgb_tensor.shape[0],rgb_tensor.shape[1]*rgb_tensor.shape[2]*rgb_tensor.shape[3])

print(flattened_features.shape)
labels = dataset.targets
construct_graph(flattened_features, labels, 'ncos')

features_file_path = "../data/usps_custom.txt"

# Write the flattened features to the file
with open(features_file_path, 'w') as file:
    for feature in flattened_features:
        file.write(' '.join([str(value.item()) for value in feature]) + '\n')

labels_path = "../data/usps_custom_labels.txt"

# Write the flattened features to the file
with open(labels_path, 'w') as labels_file:
    for label in labels:
        labels_file.write(str(label) + '\n')