from arguments import get_args_parser
from data_loader import get_data_loader, get_vocab
from model import PoetryModel
import numpy as np
import framework
import torch

if __name__ == '__main__':
    args = get_args_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_loader = get_data_loader()
    word_to_id, id_to_word = get_vocab()

    if args.model == 'Poetry':
        model = PoetryModel(args, len(word_to_id), args.layer_num)
    else:
        raise NotImplementedError

    framework.train(args, model, data_loader, word_to_id, id_to_word)
