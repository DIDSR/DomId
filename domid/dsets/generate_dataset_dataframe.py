import os
import pandas as pd
import numpy as np
path = '../../DomId/HER2/combined_train'
folders = os.listdir(path)

def total_count_images(path):
    folders = os.listdir(path)
    counter=0
    for folder in folders:
        folder_path = os.path.join(path, folder)
        images = os.listdir(folder_path)
        counter+=len(images)
        print(len(images), counter)
    return counter
def parse_machine_labels(image_names):
    machine_labels =[]
    for image in image_names:
        machine = image[-6:-4]
        machine_labels.append(machine)
    return machine_labels

number_labels = 2
N = total_count_images(path)
data =np.zeros((N, number_labels+1)).astype('str')
start = 0

for folder in folders:

    print(folder)
    print('start', start)
    folder_path = os.path.join(path, folder)
    images = os.listdir(folder_path)
    labels = [folder]*len(images)
    machine_labels = parse_machine_labels(images)

    data[start:start+len(images), :] = np.stack((images, labels, machine_labels),0).T
    start+=len(images)
    print(len(images))





dataframe =pd.DataFrame(data)
dataframe.to_csv('../HER2/combined_train/dataframe.csv', header = ['img_id', 'class', 'machine'],index=False)