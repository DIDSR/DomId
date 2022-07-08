# Domain Identification (DomId)

Deep unsupervised clustering algorithms for domain identification.

## Usage instructions

- We use Python Poetry (see <https://python-poetry.org/docs/master/#installation>) to manage dependencies and to deploy the code in this repository.
- To install the DomId package as well as the underlying libDG python package run: `poetry install`.
    - If you want to update this repository to the newest version before installing, run `git pull` (update DomId) followed by `git submodule update` (update libDG). 
- Here is a basic example to run a DL algorithm that performs supervised classification (digits) and unsupervised clustering (e.g., color) on the Color-MNIST dataset:
```
poetry run python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --debug --epos=10 --aname=m2yd --zd_dim=7 --apath=domid/algos/builder_m2yd.py
```
- Example of applying VaDE model to cluster regular MNIST:
```
poetry run python main_out.py --te_d 0 1 2 3 --tr_d 4 5 6 7 8 9 --task=mnist --debug --epos=10 --aname=vade --zd_dim=200 --d_dim=6 --apath=domid/algos/builder_vade.py
```
