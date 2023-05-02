The commit hash for the miccai code is: ```f6b98c9fe90f751eedc4bde9672ac68287146126```


# HER2 experiments
```
# VaDE base result: miccai/DomID/notebooks/2023-03-08 09:54:40.881182/
CUDA_VISIBLE_DEVICES=0 python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=vade --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=17 --dpath "../../DomId/HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005

# CDVaDE results: miccai/DomID/notebooks/2023-03-08 08:42:25.305622/
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=vade --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=17 --dpath "../../DomId/HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005 --dim_inject 3

# DEC results: miccai/DomID/notebooks/notebooks/2023-03-03 10:02:09.256589/
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=dec --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=17 --dpath "../../DomId/HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005
```


# Colored MNIST experiments: 5 digits, 5 colors
```
# VaDE base result: miccai/DomID/notebooks/2023-03-09 17:29:14.153074/
CUDA_VISIBLE_DEVICES=0 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --epos=20 --aname=vade --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=9 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn

# CDVaDE base result: miccai/DomID/notebooks/2023-03-09 17:28:58.961270/
CUDA_VISIBLE_DEVICES=1 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --epos=20 --aname=vade --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=10 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn --dim_inject 5

# DEC base result: miccai/DomID/notebooks/2023-03-09 17:37:02.633794/
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --epos=20 --aname=dec --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=10 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn
```


# Colored MNIST experiments: 10 digits, 10 colors  (command line inputs after refactoring to reproduce results)
```
# VaDE base result: miccai/DomID/notebooks/2023-03-14 11:23:06.765323/
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=20 --aname=vade --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=9 --bs 2 --lr 0.0005 --split 0.8 --prior Gaus --model cnn 

# CDVaDE base result: miccai/DomID/notebooks/2023-03-14 11:23:24.638123/
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=20 --aname=vade --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=10 --bs 2 --lr 0.0005 --split 0.8 --prior Gaus --model cnn --inject_var "color" --dim_inject 10

# DEC base result: miccai/DomID/notebooks/2023-03-09 17:37:02.633794/
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=20 --aname=dec --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=10 --bs 2 --lr 0.00005 --split 0.8 --prior Gaus --model cnn
```


# HER2 experiments (command line inputs after refactoring to reproduce results)
```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=vade --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=17 --dpath "../../DomId/HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005

poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=vade --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=17 --dpath "../../DomId/HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005 --dim_inject_y 3 --inject_var "class"

poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=dec --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=17 --dpath "../../DomId/HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005 

```


# Colored MNIST experiments: 5 digits, 5 colors (command line inputs after refactoring to reproduce results)
```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --digits_from_mnist 0 1 2 3 4 --epos=20 --aname=vade --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=7 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn

poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --digits_from_mnist 0 1 2 3 4 --epos=20 --aname=vade --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=10 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn  --dim_inject_y 5 --inject_var "color"

poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --digits_from_mnist 0 1 2 3 4 --epos=20 --aname=dec --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=10 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn
```
