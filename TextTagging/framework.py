import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn import metrics

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_metric(grnd, pred, tag2id):
    na_num = tag2id['O']
    pos_crt = 0
    pos_pre = 1e-6
    pos_tot = 1e-6

    for id in range(len(grnd)):
        if grnd[id] != na_num:
            pos_tot += 1
        if pred[id] != na_num:
            pos_pre += 1
        if grnd[id] != na_num and grnd[id] == pred[id]:
            pos_crt += 1

    rec = pos_crt/pos_tot
    prec = pos_crt/pos_pre
    f1 = 2*prec*rec/(prec+rec+1e-6)
    return f1



def evaluate(args, model, data_loader, tag2id, test=False):
    model.eval()
    dev_loss = 0.0
    labels, preds = [], []
    with torch.no_grad():
        for i, (words, mask, tags) in enumerate(data_loader):
            words = words.to(args.device)
            tags = tags.to(args.device)
            mask = mask.to(args.device)
            predict = model(words, mask)
            loss = model.log_likehood(words, tags, mask)
            dev_loss+=loss.item()
            length = []
            # store the tag
            for tag in tags.cpu().numpy():
                count = 0
                temp = []
                for j in tag:
                    if j>0:
                        count += 1
                        temp.append(j)
                length.append(count)
                labels += temp
            # store the prediction
            for idx, pred in enumerate(predict):
                preds += pred[:length[idx]]
    F1 = get_metric(labels, preds, tag2id)
    if test:
        report = metrics.classification_report(labels, preds)
        return F1, dev_loss/(len(data_loader)*args.batch_size), report
    return F1, dev_loss/(len(data_loader)*args.batch_size)

def test(args, model, data_loader, tag2id):
    model.load_state_dict(torch.load(args.save_dict))
    model.eval()
    model.to(args.device)
    test_f1, test_loss, test_report = evaluate(args, model, data_loader, tag2id, test=True)
    msg = 'Test Loss: {0:>5.2}, Test F1: {1:>6.2%}'
    print(msg.format(test_loss, test_f1))
    print("Report Matrix: ")
    print(test_report)


def train(args, model, train_loader, val_loader, test_loader, tag2id):
    start_time = time.time()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = 0.0
    bad_count = 0

    for epoch in range(args.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
        model.train()
        for i, (words, mask, tags) in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            model.to(args.device)
            words = words.to(args.device)
            mask = mask.to(args.device)
            tags = tags.to(args.device)
            loss = model.log_likehood(words, tags, mask)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10)
            optimizer.step()

        f1, dev_loss = evaluate(args, model, val_loader, tag2id)
        if f1>best_f1:
            best_f1 = f1
            bad_count = 0
            torch.save(model.state_dict(), args.save_dict)
        else:
            bad_count += 1
        msg = 'Val Loss: {0:>5.2}, Current F1: {1:>6.2%}, Best F1: {2:>6.2%}, Bad acount: {3}'
        print(msg.format(dev_loss, f1, best_f1, bad_count))
        if bad_count>args.bad_count:
            print('No optimization for a long time, auto-stopping ...')
            break
        # scheduler.step(0.8)
    test(args, model, test_loader, tag2id)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


