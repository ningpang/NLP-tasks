import os
import re
import pandas as pd
import torch
from arguments import get_args_parser
from torch.utils.data import DataLoader, Dataset
from hanziconv import HanziConv

class TextDataset(object):
    def __init__(self, args):
        self.args = args
        self.base_dir = os.path.join('./datasets', args.dataset)
        self.words, self.word_to_id = self.read_vocab()
        self.train_data, self.val_data, self.test_data = self.get_data()

    def read_vocab(self):
        vocab_dir = os.path.join(self.base_dir, 'vocab.txt')
        with open(vocab_dir) as f:
            words = [_.strip() for _ in f.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id

    def get_data(self):
        train_dir = os.path.join(self.base_dir, 'train.csv')
        val_dir = os.path.join(self.base_dir, 'dev.csv')
        test_dir = os.path.join(self.base_dir, 'test.csv')
        train_data = self.process_file(train_dir)
        val_data = self.process_file(val_dir)
        test_data = self.process_file(test_dir)

        return train_data, val_data, test_data

    def process_file(self, filename):
        f = pd.read_csv(filename, sep='\t', names=['sentence1', 'sentence2', 'label'])
        p = f['sentence1'].values[0:]
        h = f['sentence2'].values[0:]
        label = f['label'].values[0:]
        contents = []
        for i in range(len(p)):
            p_words, p_mask, p_seq_len = self.get_word_list(p[i], self.word_to_id)
            h_words, h_mask, h_seq_len = self.get_word_list(h[i], self.word_to_id)
            contents.append((p_words, p_mask, p_seq_len, h_words, h_mask, h_seq_len, int(label[i])))
        return contents

    def get_word_list(self, sentence, word_to_id):
        sentence = HanziConv.toSimplified(sentence.strip())
        regEx = re.compile('[\\W]+')
        # res = re.compile(r'([\u4e00-\u9fa5])')
        sentence = regEx.split(sentence.lower())[0]
        token_id = [word_to_id[x] for x in sentence if x in word_to_id]
        seq_len = len(token_id)

        if self.args.max_length:
            if seq_len < self.args.max_length:
                mask = [1] * len(token_id) + [0] * (self.args.max_length - len(token_id))
                token_id += [0] * (self.args.max_length - len(token_id))
            else:
                mask = [1] * self.args.max_length
                token_id = token_id[:self.args.max_length]
                seq_len = self.args.max_length

        return token_id, mask, seq_len

class data_set(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        p_words = torch.tensor([item[0] for item in data])
        p_mask = torch.tensor([item[1] for item in data])
        p_seq_len = torch.tensor([item[2] for item in data])
        q_words = torch.tensor([item[3] for item in data])
        q_mask = torch.tensor([item[4] for item in data])
        q_seq_len = torch.tensor([item[5] for item in data])
        label = torch.tensor([item[6] for item in data])
        return p_words, p_mask, p_seq_len, q_words, q_mask, q_seq_len, label

def data_loader(args, data, shuffle=True, drop_last=False, batch_size=None):
    dataset = data_set(data)
    if batch_size == None:
        batch_size = min(args.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last
    )
    return data_loader

args = get_args_parser()
dataset = TextDataset(args)

def get_train_loader():
    train_data = dataset.train_data
    return data_loader(args, train_data)

def get_val_loader():
    val_data = dataset.val_data
    return data_loader(args, val_data)

def get_test_loader():
    test_data = dataset.test_data
    return data_loader(args, test_data)



