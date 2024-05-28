# MICCAI 2023 Conference Paper

This is a summary of the steps to reproduce results presented in [Sidulova et al. 2023].
Base Model experiments are the experiments with no condition applied to the model.

[Sidulova et al. 2023] Sidulova, M., Sun, X., & Gossmann, A. (2023). Deep Unsupervised Clustering for Conditional Identification of Subgroups Within a Digital Pathology Image Set. In H. Greenspan, A. Madabhushi, P. Mousavi, S. Salcudean, J. Duncan, T. Syeda-Mahmood, & R. Taylor (Eds.), Medical Image Computing and Computer Assisted Intervention – MICCAI 2023 (Vol. 14227, pp. 666–675). Springer Nature Switzerland. <https://doi.org/10.1007/978-3-031-43993-3_64>

The same as in the `notebooks/case-study-HER2_VaDE_CDVaDE-DEC.ipynb`.


## VaDE Base Model Experiments

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=100 --aname=vade --zd_dim=250 --d_dim=3 \
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --dpath "HER2/combined_train" --split 0.8 --bs 2 \
--lr 0.00005 --prior Gaus --model cnn
```

**Notes**:
- Path to the dataset (`dpath`) needs to be adjusted depending on the location of the dataset.
- By default this and the other experiments below will use the GPU. If you wish to run the experiment on CPU use the command line argument `--nocu`.


## CVaDE: HER2 class labels

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=100 --aname=vade --zd_dim=250 --d_dim=3 \
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --dpath "HER2/combined_train" --split 0.8 --bs 2 \
--lr 0.00005 --prior Gaus --model cnn --dim_inject_y 3
```

**Note**: dimension of the injected labels (`dim_inject_y`) should be adjusted. Number of the possible class labels (e.g., for the HER2 dataset used in the paper it would be 3 - class 1, class 2, class 3)


## CVaDE: HER2 class labels + previously predicted domain labels

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=100 --aname=vade --zd_dim=250 --d_dim=3 \
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --dpath "HER2/combined_train" --split 0.8 --bs 2 \
--lr 0.00005 --prior Gaus --model cnn --dim_inject_y 6 --path_to_domain notebooks/2022-11-02\ 17\:30\:13.132561/
```

**Notes**:
- Dimension of the injected labels (`dim_inject_y`) in this case is the sum of the dimensions of the possible class labels and `d_dim` of previously predicted domains. (e.g. ).
- `path_to_domain` is the path to the previously obtained results directory and needs to be specified. Predicted domain labels should be stored within the directory `path_to_domain` in the file `domain_labels.txt`.


# CHIL 2024 Conference:

(in press) Sidulova M., Khaki S, Hegman I., Gossmann A. "Contextual unsupervised deep clustering of digital pathology dataset", Accepted in CHIL 2024.


## Experiments with Colored MNIST dataset

The following experiments are the same as in the `notebook/tutrial-MNIST-sdcn.ipynb`.

### AE pretraining

```
poetry python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --bs 128 --lr 0.0001 --zd_dim=20 --pre_tr=0 --model cnn --prior Gaus
```

### SDCN

```
poetry python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname sdcn --d_dim=10 --apath=domid/algos/builder_sdcn.py --bs 600 --lr 0.0001 --zd_dim=20 --pre_tr=2 --model cnn --prior Gaus --pre_tr_weight_path path/to/pretrained/AE/'
```

```
poetry python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname sdcn --d_dim=10 --apath=domid/algos/builder_sdcn.py --bs 600 --lr 0.0001 --zd_dim=20 --pre_tr=2 --model cnn --prior Gaus --pre_tr_weight_path 'path/to/pretrained/AE/’

```

Sample pretrained AE for colored MNIST could be found in `./notebooks/2023-08-03 09:06:05.234348_mnistcolor10_ae/`, `./notebooks/2023-08-01 13:53:39.168459_mnistcolor10_ae/`.


## Experiments with Wash U dataset


## WashU Dset

The following experiments are also set in `notebooks/case-study-WashU_AS_SDCN.ipynb`.

### AE pretraining

```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=weah --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --dpath '../../WashU-WSI-data/patches_WashU_Aperio/data/png_files/' --bs 128 --lr 0.0001 --zd_dim=100 --pre_tr=2 --model cnn --prior Gaus --tr_d_range 0 24
```

### SDCN training

```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=weah --epos=100 --aname sdcn --d_dim=6 --apath=domid/algos/builder_sdcn.py --dpath '../../WashU_with_coord/combined_training_with_coords/' --bs 1200 --lr 0.0001 --zd_dim=1000 --pre_tr=2 --model cnn --pre_tr_weight_path 'path/to/pretrained/AE' --tr_d_range 0 65 --meta_data_csv 'path/to_csv_file_that_is_generated_from_the_notebook.csv'
```

- Sample path to generated csv file: `../../WashU_with_coord/dset_WEAH.csv`, `../../WashU_with_coord/dset_WEAH_65_subjects_3_regions.csv`.
- Sample paths to pretrained AEs: `./notebooks/2023-06-30 10:38:27.253550_weah_ae/`, `./notebooks/2023-07-05 12:18:40.078628_weah_ae/`.


### VADE training

```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 --task=weah --epos=50 --aname=vade --zd_dim=500 --d_dim=6 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=17 --dpath "../../WashU_with_coord/combined_training_with_coords" --bs 4 --prior Gaus --model cnn --lr 0.0001 --tr_d_range 0 65 --meta_data_csv '../../WashU_with_coord/dset_WEAH_65_subjects_3_regions.csv'
```

### DEC training

```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 --task=weah --epos=50 --aname=dec --zd_dim=500 --d_dim=6 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=2 --dpath "../../WashU_with_coord/combined_training_with_coords" --bs 4 --prior Gaus --model cnn --lr 0.0001 --tr_d_range 0 6 --meta_data_csv '../../WashU_with_coord/dset_WEAH_65_subjects_3_regions.csv' --pre_tr_weight_path './notebooks/2023-06-30 10:38:27.253550_weah_ae/' --feat_extract ae
```

- Note: `--feat_extract` can be either "ae" of "vae", option for loading pretrained weights also in place.
