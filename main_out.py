from libdg.compos.exp.exp_main import Exp
from libdg.compos.exp.exp_cuda_seed import set_seed  # reproducibility
from libdg.arg_parser import parse_cmd_args

if __name__ == "__main__":
    import torch
    import sys
    print('torch version', torch.__version__,torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)
    print('python version', sys.version)
    args = parse_cmd_args()
    #print(args)
    #breakpoint()
    if args.task == 'mnist':
        from domid.compos.exp.exp_main import Exp
    else:
        from libdg.compos.exp.exp_main import Exp
    print(args)

    set_seed(args.seed)
    #breakpoint()
    exp = Exp(args=args)

    exp.execute()
