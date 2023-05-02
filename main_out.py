import sys

import torch
from domainlab.compos.exp.exp_cuda_seed import set_seed  # reproducibility

from domid.arg_parser import parse_cmd_args
from domid.compos.exp.exp_main import Exp
import pandas as pd
torch.cuda.empty_cache()
import os

# print('I changed the path') 

if __name__ == "__main__":
    print('torch version', torch.__version__,torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)
    print('python version', sys.version)
    args = parse_cmd_args()
    print(args)
    if os.path.exists(os.path.join(args.dpath, 'dataframe_mnist.csv')):
        os.remove(os.path.join(args.dpath, 'dataframe_mnist.csv'))

    if os.path.exists("results.csv"):
        results_df = pd.read_csv("results.csv")
    else:
        results_df = pd.DataFrame(columns=['dataset', 'model', 'seed', 'bs', 'lr', 'train_acc', 'test_acc',
                                           'similarity with vec_y', 'similarity with vec_d',
                                           'train_loss', 'test_loss'])
        results_df.to_csv("results.csv", index=False)

    set_seed(args.seed)
    exp = Exp(args=args)
    #exp.tuning()
    exp.execute()
