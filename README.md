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

## DomId usage examples (FIXME: needs updating)

- Here is a basic example to run a DL algorithm that performs supervised classification (digits) and unsupervised clustering (e.g., color) on the Color-MNIST dataset:
```
poetry run python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --debug --epos=10 --aname=m2yd --zd_dim=7 --apath=domid/algos/builder_m2yd.py --gamma_y 1
```
- Example of applying VaDE model to cluster regular MNIST:
```
poetry run python main_out.py --te_d 0 1 2 3 --tr_d 4 5 6 7 8 9 --task=mnist --debug --epos=10 --aname=vade --zd_dim=200 --d_dim=6 --apath=domid/algos/builder_vade.py
```

- Example of applying VaDE model to cluster HER2 dataset (make sure to insert dpath) 
```
poetry run python main_out.py --te_d 0 --tr_d 1 2 --task=her2 --debug --epos=30 --aname=vade --zd_dim=50 --d_dim=2 --apath=domid/algos/builder_vade.py --L=25 --pre_tr=0.80 --nocu --dpath "HER2/combined_train" --split 0.8 --bs 4 --lr 0.0005
```

- Example to run on the GPU cluster
```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --debug --epos=100 --aname=vade --zd_dim=50 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=0.75 --dpath "HER2/combined_train" --split 0.8 --bs 8 --lr 0.0005 --model cnn --prior Gaus
```

### Training conditional VaDE:
1. Labels that are going to be passed as a conditions should be stored in the "domain_labels.txt" file,
and ``` --path_to_domain``` and ``` --dim_inject_y```. Dimentions of the injected y should correspond 
to the number of domains in the "domain_label.txt". Note: only one label can be injected per experiment. 
2. According to the commandline, the dataset will return appropriate additional labels,
and in the training process those labels are going to be concatinated to the decoder input. 


## Generate documentation with Sphinx

Probably set up a separate Python virtual environment. Then run the following:

```
sh gen_doc.sh
```

## Developer hints

- To use the latest version of [DomainLab](https://github.com/marrlab/DomainLab) (rather than the version that DomId was tested with), run `git submodule update --remote`.
- By default DomId uses the master branch of DomainLab. If desired, you can set DomId to use another branch of DomainLab (replace `<branch_name>` with the name of the branch you want to use):

```
git config -f .gitmodules submodule.DomainLab.branch <branch_name>
git submodule update --remote
```

