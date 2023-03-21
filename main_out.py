import sys
sys.path.insert(0,'/home/mariia.sidulova/miccai/DomId')
sys.path.insert(0, '/home/mariia.sidulova/miccai/DomId/DomainLab')
import torch
from domainlab.compos.exp.exp_cuda_seed import set_seed  # reproducibility

from domid.arg_parser import parse_cmd_args
from domid.compos.exp.exp_main import Exp

torch.cuda.empty_cache()

# print('I changed the path') 

if __name__ == "__main__":
    print('torch version', torch.__version__,torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)
    print('python version', sys.version)
    args = parse_cmd_args()
    print(args)

    set_seed(args.seed)
    exp = Exp(args=args)
    #exp.tuning()
    exp.execute()
