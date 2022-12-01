# Base Model Experiments 
`poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --debug --epos=100 --aname=vade --zd_dim=250 --d_dim=3
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --nocu --dpath "HER2/combined_train" --split 0.8 --bs 2 --lr 0.00005 --prio Gaus --model cnn `
# CVaDE: class labels 
`poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --debug --epos=100 --aname=vade --zd_dim=250 --d_dim=3
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --nocu --dpath "HER2/combined_train" --split 0.8 --bs 2 --lr 0.00005 --prio Gaus --model cnn --dim_inject_y 3`
# CVaDE: class labels + prior domain labels 
`poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --debug --epos=100 --aname=vade --zd_dim=250 --d_dim=3
--apath=domid/algos/builder_vade.py --L=5 --pre_tr=20 --nocu --dpath "HER2/combined_train" --split 0.8 --bs 2 --lr 0.00005 --prio Gaus 
--model cnn --dim_inject_y 3 --path_to_domain notebooks/2022-11-02 17:30:13.132561/`
