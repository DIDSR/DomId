Here is the summary of steps to reproduce results presented in the paper. 
Base Model experiments are the experiments with no condition applied to the model. 


# Base Model Experiments

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=100 --aname=vade --zd_dim=250 --d_dim=3 \
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --dpath "HER2/combined_train" --split 0.8 --bs 2 \
--lr 0.00005 --prior Gaus --model cnn
```

**Notes**:
- Path to the dataset (`dpath`) needs to be adjusted depending on the location of the dataset.
- By default this and the other experiments below will use the GPU. If you wish to run the experiment on CPU use the command line argument `--nocu`.


# CVaDE: HER2 class labels 

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=100 --aname=vade --zd_dim=250 --d_dim=3 \
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --dpath "HER2/combined_train" --split 0.8 --bs 2 \
--lr 0.00005 --prior Gaus --model cnn --dim_inject_y 3
```

**Note**: dimension of the injected labels (`dim_inject_y`) should be adjusted. Number of the possible class labels 
(e.g., for the HER2 dataset used in the paper it would be 3 - class 1, class 2, class 3)

# CVaDE: HER2 class labels + previously predicted domain labels 

```
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=100 --aname=vade --zd_dim=250 --d_dim=3 \
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --dpath "HER2/combined_train" --split 0.8 --bs 2 \
--lr 0.00005 --prio Gaus --model cnn --dim_inject_y 6 --path_to_domain notebooks/2022-11-02 17:30:13.132561/
```

**Note**: dimension of the injected labels (`dim_inject_y`) in this case is the sum of the dimensions of the possible class labels and `d_dim` of previously predicted domains. (e.g. ). Also, `path_to_domain` is the path to the preciously obtained results needs to be specified. Predicted domain labels should be stored in 'domain_labels.txt'.

# Analyzing data

If you'd like to run analysis for each of the produced cluters/domains please use the following notebook:
`notebooks\HER2_machine_clustering_3_cluster.ipnb`.

Results from each of the experiments is saved in the notebook directory. Inside each experiment directory 
there is an `argument.txt` file which contains following information: model name, encoder/decoder structure, prior distribution, 
number of latent features, test domains, train domains, L, learning rate, batch size, number of pretrain epochs, and total number of epochs.

# TensorBoard

Live feed of loss calculations can be observed on the tensorboard, which is stored in the `debug` directory. Furthermore, 
input vs reconstructed images are also vizualized there. To access tensorboard, one can type in the terminal 
`tensorboard --logdir debug`.

