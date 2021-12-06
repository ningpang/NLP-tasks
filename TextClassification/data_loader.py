import os
import torch
from arguments import get_args_parser
from torch.utils.data import DataLoader, Dataset


class TextDataset(object):
    def __init__(self, args):
        self.args = args
        self.base_dir = os.path.join('./datasets', args.dataset)
        self.words, self.word_to_id = self.read_vocab()
        self.classes, self.cls_to_id = self.read_classes()
        self.train_data, self.val_data, self.test_data = self.get_data()

    def read_vocab(self):
        vocab_dir = os.path.join(self.base_dir, 'vocab.txt')
        with open(vocab_dir) as f:
            words = [_.strip() for _ in f.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id

    def read_classes(self):
        class_dir = os.path.join(self.base_dir, 'class.txt')
        with open(class_dir) as f:
            classes = [_.strip() for _ in f.readlines()]
        cls_to_id = dict(zip(classes, range(len(classes))))
        return classes, cls_to_id

    def get_data(self):
        train_dir = os.path.join(self.base_dir, 'train.txt')
        val_dir = os.path.join(self.base_dir, 'val.txt')
        test_dir = os.path.join(self.base_dir, 'test.txt')
        train_data = self.process_file(train_dir, self.word_to_id, self.cls_to_id)
        val_data = self.process_file(val_dir, self.word_to_id, self.cls_to_id)
        test_data = self.process_file(test_dir, self.word_to_id, self.cls_to_id)

        return  train_data, val_data, test_data

    def process_file(self, filename, word_to_id, cls_to_id, max_length=600):
        contents = []
        with open(filename) as f:
            for line in f:
                try:
                    label, content = line.strip().split('\t')
                    if content:
                        token_id = [word_to_id[x] for x in content if x in word_to_id]
                        seq_len = len(token_id)
                        if max_length:
                            if len(token_id)<max_length:
                                mask = [1]*len(token_id)+[0]*(max_length-len(token_id))
                                token_id += [0]*(max_length-len(token_id))
                            else:
                                mask = [1]*max_length
                                token_id = token_id[:max_length]
                                seq_len = max_length
                        label_id = cls_to_id[label]
                        contents.append((token_id, label_id, seq_len, mask))
                except:
                    pass
        return contents

class data_set(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        content = torch.tensor([item[0] for item in data])
        label = torch.tensor([item[1] for item in data])
        seq_len = torch.tensor([item[2] for item in data])
        mask = torch.tensor([item[3] for item in data])
        return content, label, seq_len, mask

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

def get_class_list():
    class_list = dataset.classes
    return class_list


