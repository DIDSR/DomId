# Domain Identification (DomId)

Deep unsupervised clustering algorithms for domain identification.

## Initial setup instructions
1. Clone this repository, e.g.,
```
git clone https://github.com/agisga/DomId.git
```
2. (Optional) Switch to a branch, e.g.,
```
git checkout Mariia-DomID
```
3. Initialize the DomainLab submodule:
    - Enter the DomainLab subfolder: `cd DomainLab`
    - `git submodule init`
    - `git submodule update` to fetch all the data from DomainLab and check out the appropriate commit listed in DomId configuration. Alternatively, to check out the latest commit of DomainLab use `git submodule update --remote`.
    - Go back to the DomId directory: `cd ..`
4. Install `DomId` and `DomainLab` packages with:
```
poetry install
```

## Usage instructions

- We use Python Poetry (see <https://python-poetry.org/docs/master/#installation>) to manage dependencies and to deploy the code in this repository.
- To install the DomId package as well as the underlying DomainLab python package run: `poetry install`.
    - If you want to update this repository to the newest version before installing, run `git pull` (update DomId) followed by `git submodule update --remote` (update DomainLab). 
- Here is a basic example to run a DL algorithm that performs supervised classification (digits) and unsupervised clustering (e.g., color) on the Color-MNIST dataset:
```
poetry run python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --debug --epos=10 --aname=m2yd --zd_dim=7 --apath=domid/algos/builder_m2yd.py
```
- Example of applying VaDE model to cluster regular MNIST:
```
poetry run python main_out.py --te_d 0 1 2 3 --tr_d 4 5 6 7 8 9 --task=mnist --debug --epos=10 --aname=vade --zd_dim=200 --d_dim=6 --apath=domid/algos/builder_vade.py
```

- Example of applying VaDE model to cluster HER2 dataset (make sure to insert dpath)
```
poetry run python main_out.py --te_d 0 --tr_d 1 2 --task=her2 --debug --epos=30 --aname=vade --zd_dim=50 --d_dim=2 --apath=domid/algos/builder_vade.py --L=25 --pre_tr=0.80 --nocu --dpath "HER2/combined_train" --split 0.8 --bs 4 --lr 0.0005
```
conda activate env_domainid   # change environment which
git config -f .gitmodules submodule.DomainLab.branch python-3-10
git submodule update --remote   # without --remote, it is not the up to date version!!
cd DomainLab
git log   # check if everything is updated with respect to the server
```
