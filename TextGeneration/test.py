from arguments import get_args_parser
from model import PoetryModel
import numpy as np
import framework
import torch
import json
import os

if __name__ == '__main__':
    args = get_args_parser()
    base_dir = os.path.join('./datasets', args.dataset)
    word_to_id = json.load(open(os.path.join(base_dir, 'word2id.json'), 'r'))
    id_to_word = {id: word for word, id in list(word_to_id.items())}

    if args.model == 'Poetry':
        model = PoetryModel(args, len(word_to_id), args.layer_num)
    else:
        raise NotImplementedError

    title = '春江花月夜凉如水'
    framework.test(args, model, title, word_to_id, id_to_word)
