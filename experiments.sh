poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=20 --aname=vade --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=9 --bs 2 --lr 0.0005 --split 0.8 --prior Gaus --model cnn
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=20 --aname=vade --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=10 --bs 2 --lr 0.0005 --split 0.8 --prior Gaus --model cnn --inject_var "color" --dim_inject 10
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 5 6 7 8 9 --task=mnistcolor10 --epos=20 --aname=dec --zd_dim=20 --d_dim=10 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=10 --bs 2 --lr 0.00005 --split 0.8 --prior Gaus --model cnn
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=vade --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=17 --dpath "../../HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=vade --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=17 --dpath "../../HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005 --dim_inject_y 3 --inject_var "class"
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 --task=her2 --epos=50 --aname=dec --zd_dim=500 --d_dim=3 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=17 --dpath "../../HER2/combined_train" --bs 4 --prior Gaus --model cnn --lr 0.000005
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --digits_from_mnist 0 1 2 3 4 --epos=20 --aname=vade --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=7 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --digits_from_mnist 0 1 2 3 4 --epos=20 --aname=vade --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_vade.py --L=5 --pre_tr=10 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn  --dim_inject_y 5 --inject_var "color"
poetry run python main_out.py --te_d 0 --tr_d 0 1 2 3 4 --task=mnistcolor10 --digits_from_mnist 0 1 2 3 4 --epos=20 --aname=dec --zd_dim=20 --d_dim=5 --apath=domid/algos/builder_dec.py --L=5 --pre_tr=10 --bs 2 --lr 0.005 --split 0.8 --prior Gaus --model cnn

