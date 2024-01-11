import sys

sys.path.insert(0, "/home/mariia.sidulova/scdn/DomId")
sys.path.insert(0, "/home/mariia.sidulova/scdn/DomId/DomainLab")

import pandas as pd
import torch
from domainlab.compos.exp.exp_cuda_seed import set_seed  # reproducibility

from domid.arg_parser import parse_cmd_args
from domid.compos.exp.exp_main import Exp

torch.cuda.empty_cache()
import os

# print('I changed the path')

if __name__ == "__main__":
    print("torch version", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)
    print("python version", sys.version)
    args = parse_cmd_args()
    print(args)
    # try:
    #     assert len(args.tr_d) == args.d_dim
    # except AssertionError:
    #     print("The number of training domains does not match the dimension of the domain space.")
    #     sys.exit(1)

    if os.path.exists(os.path.join(args.dpath, "dataframe_mnist.csv")):
        os.remove(os.path.join(args.dpath, "dataframe_mnist.csv"))

    set_seed(args.seed)
    exp = Exp(args=args)
    # exp.tuning()
    exp.execute()
