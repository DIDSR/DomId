Git commit hash: 


# Experiments with MNIST dataset

* AE pretraining: 
```
python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnist --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --bs 256 --lr 0.0001 --zd_dim=20 --pre_tr=0 --model cnn --prior Guas
```
* SDCN:
```
python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnist --epos=10 --aname sdcn --d_dim=10 --apath=domid/algos/builder_sdcn.py --bs 2500 --lr 0.0001 --zd_dim=20 --pre_tr=2 --model cnn --prio Gaus --pre_tr_weight_path 'path/to/pretrained/AE/'
```
Sample path to the pretrained AE is inside mariia.sidulova/scdn/DomId/notebooks: ```./notebooks/2023-08-01 12:45:45.645754_mnist_ae/```
Note: bs size should be adjusted based on the experimenta setup. 

# Experiments with color MNIST dataset
* AE pretraining: 
```
AE pretraining: 
python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --bs 128 --lr 0.0001 --zd_dim=20 --pre_tr=0 --model cnn --prior Gaus
```
* SDCN:
```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname sdcn --d_dim=10 --apath=domid/algos/builder_sdcn.py --bs 600 --lr 0.0001 --zd_dim=20 --pre_tr=2 --model cnn --prior Gaus --pre_tr_weight_path path/to/pretrained/AE/'
```
```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname sdcn --d_dim=10 --apath=domid/algos/builder_sdcn.py --bs 600 --lr 0.0001 --zd_dim=20 --pre_tr=2 --model cnn --prior Gaus --pre_tr_weight_path 'path/to/pretrained/AE/’

```
Sample pretrained AE for colored MNIST could be found: ```./notebooks/2023-08-03 09:06:05.234348_mnistcolor10_ae/```, ```./notebooks/2023-08-01 13:53:39.168459_mnistcolor10_ae/```

# Experiments with Wash U dataset

## Obtaining the csv file for the dataset
To execute the code, you need first acquire the CSV file containing metadata. Within the designated folder, the list of images should determine the sequence in which images are loaded into the Dataloaders.
For MNIST, this process is automated and can be found in 'dset/a_mnist_dset.py'. 
However, for the Wash U dataset (pathology dataset), you'll need to execute 'notebooks/WashU_dset.py'. This notebook generates the CSV file containing patch paths and their corresponding metadata.

The current set of experiments is based on the data that has been extracted from both annotated and outside of annotated regions. The patches are located in the following directory: 
```WashU_with_coord/combined_training_with_coords/```.

## WashU Dset
* AE pretraining: 
```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=weah --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --dpath '../../WashU-WSI-data/patches_WashU_Aperio/data/png_files/' --bs 128 --lr 0.0001 --zd_dim=100 --pre_tr=2 --model cnn --prior Gaus --tr_d_range 0 24
```

* SDCN training: 

```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=weah --epos=100 --aname sdcn --d_dim=6 --apath=domid/algos/builder_sdcn.py --dpath '../../WashU_with_coord/combined_training_with_coords/' --bs 1200 --lr 0.0001 --zd_dim=1000 --pre_tr=2 --model cnn --pre_tr_weight_path 'path/to/pretrained/AE' --tr_d_range 0 65 --meta_data_csv 'path/to_csv_file_that_is_generated_from_the_notebook.csv'
```
Sample path to generated csv file: ```../../WashU_with_coord/dset_WEAH.csv ```, ```../../WashU_with_coord/dset_WEAH_65_subjects_3_regions.csv ```
Sample paths to pretrained AEs: ```./notebooks/2023-06-30 10:38:27.253550_weah_ae/```, ```./notebooks/2023-07-05 12:18:40.078628_weah_ae/ ```
