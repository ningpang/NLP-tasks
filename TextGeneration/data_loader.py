import os
import json
import re
import torch
from arguments import get_args_parser
from torch.utils.data import DataLoader, Dataset


class TextDataset(object):
    def __init__(self, args):
        self.args = args
        self.base_dir = os.path.join('./datasets', args.dataset)
        self.raw_data = self.get_raw_data()
        self.word_to_id, self.id_to_word = self.build_vocab(self.raw_data)

        with open(os.path.join(self.base_dir, 'word2id.json'), 'w', encoding='utf-8') as f:
            json.dump(self.word_to_id, f)

        self.train_data = self.get_data()

    def build_vocab(self, data):
        words = {c for line in data for c in line}
        word_to_id = {word: id for id, word in enumerate(words)}
        word_to_id['<EOP>'] = len(word_to_id)
        word_to_id['<START>'] = len(word_to_id)
        word_to_id['</s>'] = len(word_to_id)
        id_to_word = {id: word for word, id in list(word_to_id.items())}
        return word_to_id, id_to_word

    def get_data(self):
        contents = []
        for i in range(0, len(self.raw_data)):
            peotry = ['<START>']+list(self.raw_data[i])+['<EOP>']
            peotry_id = [self.word_to_id[word] for word in peotry]
            peotry_pad = self.pad_peotry(peotry_id, pad='pre', trc='post')
            contents.append(peotry_pad)
        return contents

    def get_raw_data(self):
        contents = []
        for filename in os.listdir(self.base_dir):
            if filename.startswith(self.args.category):
                contents.extend(self.process_file(os.path.join(self.base_dir, filename)))
        return contents

    def process_file(self, filename):
        def sentence_parse(sentence):
            """对文本进行处理，取出脏数据"""
            # 去掉括号中的部分
            # para = "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。"
            result, number = re.subn("（.*）", "", sentence)
            result, number = re.subn("{.*}", "", result)
            result, number = re.subn("《.*》", "", result)
            result, number = re.subn("《.*》", "", result)
            result, number = re.subn("[\]\[]", "", result)
            # 去掉数字
            r = ""
            for s in result:
                if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                    r += s
            # 处理两个句号为1个句号
            r, number = re.subn("。。", "。", r)
            # 返回预处理好的文本
            return r
        poetrys = []
        data = json.load(open(filename, 'r'))
        for poetry in data:
            if self.args.author is not None and poetry['author'] != self.args.author:
                continue
            flag = False
            for sen in poetry['paragraphs']:
                s_sen = re.split("[，！。]", sen)
                for tr in s_sen:
                    if self.args.constrain is not None and len(tr) != self.args.constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue

            pdata = ''
            for sen in poetry['paragraphs']:
                pdata += sen
            pdata = sentence_parse(pdata)
            if pdata != '' and len(pdata)>1:
                poetrys.append(pdata)

        return poetrys

    def pad_peotry(self, word_id, pad, trc):
        if len(word_id) < self.args.max_length:
            pad_length = self.args.max_length - len(word_id)
            if pad == 'pre':
                word_id = [self.word_to_id['</s>']]*pad_length + word_id
            elif pad == 'post':
                word_id = word_id + [self.word_to_id['</s>']]*pad_length
        else:
            if trc == 'pre':
                word_id = word_id[-self.args.max_length:]
            else:
                word_id = word_id[:self.args.max_length]
        return word_id

class data_set(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        words = torch.tensor([item for item in data])
        return words

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

def get_data_loader():
    train_data = dataset.train_data
    return data_loader(args, train_data)

def get_vocab():
    return dataset.word_to_id, dataset.id_to_word