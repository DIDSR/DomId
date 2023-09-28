Git commit hash: 
#  MNIST experiments
AE pretraining: 
```
python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnist --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --bs 256 --lr 0.0001 --zd_dim=20 --pre_tr=0 --model cnn
```
SDCN:
```
python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnist --epos=10 --aname sdcn --d_dim=10 --apath=domid/algos/builder_sdcn.py --bs 2500 --lr 0.0001 --zd_dim=20 --pre_tr=2 --model cnn --pre_tr_weight_path './notebooks/2023-08-01 12:45:45.645754_mnist_ae/'
```

# Color MNIST experiments
AE pretraining: 
```
python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --bs 128 --lr 0.0001 --zd_dim=20 --pre_tr=0 --model cnn --prior Gaus
```
SDCN:
```
CUDA_VISIBLE_DEVICES=2 python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=100 --aname sdcn --d_dim=10 --apath=domid/algos/builder_sdcn.py --bs 600 --lr 0.0001 --zd_dim=20 --pre_tr=2 --model cnn --prior Gaus --pre_tr_weight_path './notebooks/2023-08-01 13:53:39.168459_mnistcolor10_ae/'
```
# WashU Dset
AE pretraining: 
```
python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=weah --epos=100 --aname ae --d_dim=10 --apath=domid/algos/builder_AE.py --dpath '../../WashU-WSI-data/patches_WashU_Aperio/data/png_files/' --bs 128 --lr 0.0001 --zd_dim=100 --pre_tr=2 --model cnn --prior Gaus --tr_d_range 0 24
```


