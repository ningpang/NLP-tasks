from arguments import get_args_parser
from data_loader import get_test_loader, get_word_tag
from model import LSTMCRF
import torch
import numpy as np
import framework

if __name__ == '__main__':
    args = get_args_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_loader = get_test_loader()
    word2id, tag2id = get_word_tag()

    if args.model == 'LSTMCRF':
        model = LSTMCRF(args, word2id, tag2id)
    else:
        raise NotImplementedError

    framework.test(args, model, test_loader, tag2id)




