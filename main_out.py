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

    assert len(args.tr_d) == args.d_dim
    if os.path.exists(os.path.join(args.dpath, 'dataframe_mnist.csv')):
        os.remove(os.path.join(args.dpath, 'dataframe_mnist.csv'))


    set_seed(args.seed)
    exp = Exp(args=args)
    #exp.tuning()
    exp.execute()
