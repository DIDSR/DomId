# Domain Identification (DomId)

![GH Actions CI ](https://github.com/agisga/DomId/actions/workflows/ci.yml/badge.svg)

Deep unsupervised clustering algorithms for domain identification.

## Installation

0. Prerequisites:
    - This Python package uses [Poetry](https://python-poetry.org/) for dependency management and various package development workflows. To install Poetry see: <https://python-poetry.org/docs/#installation>.
    - A workflow without Poetry is also possible, but it is currently not recommended. For a workflow without Poetry, dependencies can be installed from the `requirements.txt` files in the `DomId` and `DomainLab` folders. Note that in order to run `DomId` without installation via Poetry, you also will have to manually add `DomainLab` to the python path.
1. Clone this repository, e.g.,
```
git clone https://github.com/agisga/DomId.git
```
2. Set up the [DomainLab](https://github.com/marrlab/DomainLab) submodule:
    - Enter the `DomId` directory, then run the following commands.
    - `git submodule init`
    - `git submodule update` to fetch all the data from DomainLab and check out the appropriate commit listed in DomId configuration.
3. Install `DomId` and `DomainLab` packages as well as all dependencies with:
```
poetry install
```

*Note*: DomId will be published to PyPI in the near future, and the installation will be as easy as `pip install domid`.

## Usage

### VaDE model

The deep unsupervised clustering model VaDE has been proposed in [Jiang et al. 2017].

*[Jiang et al. 2017]* Jiang, Zheng, Tan, Tang, and Zhou, "Variational deep embedding: An unsupervised and generative approach to clustering," in IJCAI, 2017. <http://arxiv.org/abs/1611.05148>

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


### Custom datasets

To apply a deep clustering model, such as VaDE, to a custom (e.g., your own) dataset one needs to define a dataset class and a task class. For example, for the "HER2" dataset used in [Sidulova et al 2023] the respective python files are `domid/dsets/dset_her2.py` and `domid/tasks/task_her2.py`. Finally, the defined new task should be added in the chain defined in `domid/tasks/zoo_tasks.py`. For example, with the defined "HER2" dataset and task, the following command would apply the VaDE model to the "HER2" dataset:

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --debug --epos=30 --aname=vade --zd_dim=250 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=10 --dpath "/path/to/HER2/combined_train" --split 0.8 --bs 2 --lr 0.00005 --prio Gaus --model cnn
```

### CVaDE model

The Conditional VaDE model (CVaDE) has been proposed in [Sidulova et al. 2023].

*[Sidulova et al. 2023]* Sidulova, Sun, Gossmann, "DEEP UNSUPERVISED CLUSTERING FOR CONDITIONAL IDENTIFICATION OF SUBGROUPS WITHIN A DIGITAL PATHOLOGY IMAGE SET," in review, 2023.

#### Training CVaDE:

- Additional labels (for example previously predicted cluster assignments or known subgroups) that are going to be passed as a conditions should be stored in a `domain_labels.txt` file, and the command line arguments `--path_to_domain` and `--dim_inject_y` should be specified when executing the CVaDE model. Note that the dimensionality of the injected label should correspond to the number of domains in `domain_label.txt`. Note: only one label can be injected per experiment in this way.
- According to the command line, the pytorch dataset object / data generator will return appropriate additional labels, and in the training process those labels are going to be concatinated to the decoder input.
- Furthermore, the y label specified in the pytorch dataset class is injected, either as the only conditioning variable or in addition to the labels from `domain_labels.txt`, if `--path_to_domain` and `--dim_inject_y` are set appropriately. That is, two types of "labels" can be passed to the model at the same time to condition the clustering process.
- Example command line calls are given in `README_paper.md` in this code repository.

### Simultaneous unsupervised clustering and supervised classification

#### M2YD model

The M2YD model is a rather simple (toy) DL model implemented in `DomId`, that simultaneously performs a supervised classification task (e.g., digit labels in color-MNIST) and an unsupervised clustering task (e.g., cluster the colors in color-MNIST).
Here is a basic example to run the M2YD model on the Color-MNIST dataset:

```
poetry run python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --debug --epos=10 --aname=m2yd --zd_dim=7 --apath=domid/algos/builder_m2yd.py --gamma_y 1
```
