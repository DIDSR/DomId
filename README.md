# Domain Identification (DomId): A suite of deep unsupervised clustering algorithms

![GH Actions CI ](https://github.com/DIDSR/DomId/actions/workflows/ci.yml/badge.svg)

DomId is a Python package offering a PyTorch-based suite of unsupervised deep clustering algorithms. The primary goal is to identify subgroups that have not been previously annotated within image datasets.

Some of the implemented models are the Variational Deep Embedding (VaDE) model [Jiang et al., 2017], Conditionally Decoded Variational Deep Embedding (CDVaDE) [Sidulova et al., 2023], Deep Embedding Clustering (DEC) [Xie et al., 2016], Structural Deep Clustering Network (SDCN) [Bo et al., 2020].

These clustering algorithms include a feature extractor component, which can be either an Autoencoders (AE) or a Variational Autoencoder (VAE). The package provides multiple AE and VAE architectures to choose from and includes instructions for extending the package with custom neural network architectures or clustering algorithms.

Ready-to-use experiment tutorials in Jupyter notebooks are available for both the MNIST dataset and a digital pathology dataset.

By adopting a highly modular design, the codebase prioritizes straightforward extensibility, so that new models, datasets or tasks can be added with ease.
The software design of DomId follows the design principles of [DomainLab](https://github.com/marrlab/DomainLab), which is a modular Python package for training domain invariant neural networks and has been used to develop DomId.

## Installation

0. Prerequisites:
    - This Python package uses [Poetry](https://python-poetry.org/) for dependency management and various package development workflows. To install Poetry see: <https://python-poetry.org/docs/#installation>.
    - A workflow without Poetry is also possible, but it is currently not recommended. For a workflow without Poetry, dependencies can be installed from the `requirements.txt` file.
1. Clone this repository, e.g.,
```
git clone https://github.com/agisga/DomId.git
```
3. Install the `DomId` package and all its dependencies with:
```
poetry install
```

## Usage

The following examples demonstrate how to use the DomId Python package directly from your command line. You can also leverage its API within your Python scripts or notebooks for greater flexibility. For in-depth tutorials and case studies, please refer to the `notebooks` directory.

### VaDE model

The deep unsupervised clustering model VaDE has been proposed in [1].

#### Applying VaDE to MNIST

For examle, to cluster digits 0-4 of the MNIST dataset with the VaDE model proposed in [Jiang et al. 2017] using CNN encoder and decoder architectures with a 200-dimensional latent space:

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnist --epos=10 --aname=vade --zd_dim=200 --d_dim=5 --apath=domid/algos/builder_vade.py --split 0.8 --bs 100 --pre_tr 5 --lr 0.00005 --model cnn
```

The output metrics and confusion matrices assess how well the identified clusters correspond to the digit labels:

```
(...)
Epoch 10. ELBO loss
pi:
[0.14428806 0.22498095 0.1877629  0.24480581 0.19816224]
epoch: 10
pooled train clustering acc:  0.7213934426229508
[[3407    0   13    2    2]
 [   0 5203   31   61   20]
 [  18  104 1243 1293    7]
 [1266   24 3204 3299  178]
 [  34   46  265  230 4450]]
clustering validation acc:  0.7236065573770492
[[ 848    0    1    6    0]
 [   0 1305   13    8    8]
 [   7   26  340  311    3]
 [ 316    4  816  813   44]
 [   9   10   53   51 1108]]
```

#### Applying VaDE to Color-MNIST

- To apply VaDE to the color-MNIST dataset:

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --debug --epos=20 --pre_tr=10 --aname=vade --zd_dim=50 --d_dim=10 --apath=domid/algos/builder_vade.py --model cnn
```

The output metrics and confusion matrices assess how well the identified clusters correspond to the different colors:

```
(...)
Epoch 20. ELBO loss
pi:
[0.09958664 0.09956316 0.09337884 0.0995345  0.04938212 0.09954792
 0.2096352  0.09956467 0.09960268 0.05020422]
epoch: 20
pooled train clustering acc:  0.7561666666666667
[[600   0   0   0   0   0   0   0   0   0]
 [  0 600   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 600   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0 262]
 [  0   0   0   0   0 599   0   0   0   0]
 [  0   0 600   0 600   1 600   0   0   0]
 [  0   0   0   0   0   0   0 600   0   0]
 [  0   0   0   0   0   0   0   0 600   0]
 [  0   0   0   0   0   0   0   0   0 338]]
clustering validation acc:  0.7568333333333334
[[600   0   0   0   0   0   0   0   0   0]
 [  0 600   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0 600   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0 259]
 [  0   0   0   0   0 600   0   0   0   0]
 [  0   0 600   0 600   0 600   0   0   0]
 [  0   0   0   0   0   0   0 600   0   0]
 [  0   0   0   0   0   0   0   0 600   0]
 [  0   0   0   0   0   0   0   0   0 341]]
```

### DEC model
To apply a DEC model, that is described in the paper [3], to MNIST dataset:

``` 
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnist10 --debug --epos=20 --pre_tr=10 --aname=dec --zd_dim=50 --d_dim=10 --apath=domid/algos/builder_dec.py --model cnn
```
Note: there is no conditioning on the labels in the DEC model.


### Custom datasets

To apply a deep clustering model, such as VaDE, to a custom (e.g., your own) dataset one needs to define a dataset class and a task class. For example, for the "HER2" dataset used in [xxxx 2023] the respective python files are `domid/dsets/dset_her2.py` and `domid/tasks/task_her2.py`. Finally, the defined new task should be added in the chain defined in `domid/tasks/zoo_tasks.py`. For example, with the defined "HER2" dataset and task, the following command would apply the VaDE model to the "HER2" dataset:

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --debug --epos=30 --aname=vade --zd_dim=250 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=10 --dpath "/path/to/HER2/combined_train" --split 0.8 --bs 2 --lr 0.00005 --prio Gaus --model cnn
```

### CDVaDE model

The Conditionally Decoded VaDE model (CDVaDE) has been proposed in [4].


#### Training CVaDE:
In order to train the CVaDE model, the following command can be used:


- Generate csv file for the dataset, with the following columns: `image_id`, `label`, `domain_label`.
The `image_id` column should contain the path to the image, the `label` column should contain the label of the image, 
and the `domain_label` column should contain the domain label of the image. The `domain_label` column is optional,
and if it is not present, the CVaDE model will be trained without domain labels. 
Note: that the name of the columns could be different, but the values can only be integers. 
- The csv file should be stored in the root `data` directory (same as `zd_path`).

- If csv file is in a differente directory the path can be specified in the `--csv_file` command line argument.
- Injected variable and the dimentions of the injected variable should be specified in the `--inject_var` and `--dim_inject_y` command line argument.
- The `--inject_var` argument should be a string matching the name of one of the columns in the generated dataframe. 
- The `--dim_inject_y` argument should be an integer specifying the number of unique values in the column specified in the `--inject_var` argument.

For example, to train the CVaDE model on the Color-MNIST dataset with injection of color labels, the following command can be used:

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --epos=20 --aname=vade --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=9 --bs 2 --lr 0.00005 --split 0.8 --prior Gaus --model cnn --inject_var "color" --dim_inject_y 5

```
For color-MNIST csv file is generated automatically in the `a_dset_mnist_color_rgb_solo.py`. 
For other datasets, the csv file should be generated manually. 

Assuming that there is a generated csv for HER2 dataset in the `"../HER2/combined_train/"` (note: the code for generating csv for HER2 can be found in `dset` folder), CDVaDE can be trained with the following command:
The injected variable is the class of the image, and the number of unique values in the class column is 3.
```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=20 --aname=vade --zd_dim=20 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=9 --bs 2 --lr 0.00005 --split 0.8 --prior Gaus --model cnn --inject_var "class" --dim_inject_y 3 --dpath "../HER2/combined_train/"
```

### SDCN, and modified SDCN for WSI data

SDCN is a deep neural network model that combines GCN and AE architectures for the purpose of unsupervised clustering.[5]

However, original SDCN model faces significant scalability challenges that hinder its deployment in digital pathology, 
particularly when dealing with whole-slide digital pathology images (WSI), which are typically of gigapixel size or larger.
This limitation arises from SDCN need for constructing a graph on the entire dataset and the imperative to process all data in a single
batch during training. To overcome this issue, we propose batching strategy to the SDCN training process and introduce 
a novel batching approach tailored specifically for WSI data.[6]


```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=30 --aname=ae --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_AE.py --L=5 --pre_tr=2 --bs 600 --lr 0.0001 --model cnn --nocu
```

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=30 --aname=sdcn --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_sdcn.py --L=5 --pre_tr=2 --bs 600 --lr 0.0001 --model cnn --pre_tr_weight_path 'path/to/pretrained_weights/folder/' --nocu

```

### Simultaneous unsupervised clustering and supervised classification
#### M2YD model

The M2YD model is an experimental (toy) DL model implemented in `DomId`, that simultaneously performs a supervised classification task (e.g., digit labels in color-MNIST) and an unsupervised clustering task (e.g., cluster the colors in color-MNIST).
Here is a basic example to run the M2YD model on the Color-MNIST dataset:

```
poetry run python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --debug --epos=10 --aname=m2yd --zd_dim=7 --apath=domid/algos/builder_m2yd.py --gamma_y 1
```


# References

[1] Jiang, Zhuxi, et al. "Variational deep embedding: An unsupervised and generative approach to clustering." IJCAI 2017. (<https://arxiv.org/abs/1611.05148>)

[2] Kingma, Welling. "Auto-encoding variational bayes." ICLR 2013. (<https://arxiv.org/abs/1312.6114>) 

[3] Xie, Girshick, Farhadi. "Unsupervised Deep Embedding for Clustering Analysis." ICML 2016. (<http://arxiv.org/abs/1511.06335>)

[4] Sidulova, Sun, Gossmann. "Deep Unsupervised Clustering for Conditional Identification of Subgroups Within a Digital Pathology Image Set." MICCAI, 2023. (<https://link.springer.com/chapter/10.1007/978-3-031-43993-3_64>)

[5] Bo, Deyu, et al. "Structural deep clustering network." Proceedings of the web conference 2020. 2020. (<https://doi.org/10.1145/3366423.3380214>)

[6] Sidulova, Kahaki, Hagemann, Gossmann. "Contextual unsupervised deep clustering in digital pathology." CHIL 2024.
