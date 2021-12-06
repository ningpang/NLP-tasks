from arguments import get_args_parser

from data_loader import get_test_loader
from model import ESIM
import framework
import torch
import numpy as np

if __name__ == '__main__':
    args = get_args_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_loader = get_test_loader()
    #
    if args.model == 'ESIM':
        model = ESIM(args)
    else:
        raise NotImplementedError

    framework.test(args, model, test_loader)




