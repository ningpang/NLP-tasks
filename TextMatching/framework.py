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

def evaluate(args, model, data_loader, test=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    dev_loss = 0.0
    with torch.no_grad():
        for i, (p_words, p_mask, _, h_words, h_mask, _, label) in enumerate(data_loader):
            p_words = p_words.to(args.device)
            p_mask = p_mask.to(args.device)
            h_words = h_words.to(args.device)
            h_mask = h_mask.to(args.device)
            label = label.to(args.device)
            logits, probs = model(p_words, p_mask, h_words, h_mask)
            loss = criterion(logits, label)
            dev_loss+=loss.item()

            label = label.data.cpu().numpy()
            predict = torch.max(probs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, dev_loss/len(data_loader), confusion
    return acc, dev_loss/len(data_loader)

def test(args, model, data_loader):
    model.load_state_dict(torch.load(args.save_dict))
    model.eval()
    model.to(args.device)
    test_acc, test_loss, test_confusion = evaluate(args, model, data_loader, test=True)
    msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Confusion Matrix")
    print(test_confusion)


def train(args, model, train_loader, val_loader, test_loader):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
    #                                                        factor=0.85, patience=0)
    best_acc = 0.0
    bad_count = 0

    for epoch in range(args.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
        model.train()
        for i, (p_words, p_mask, _, h_words, h_mask, _, label) in enumerate(train_loader):
            model.zero_grad()
            model.to(args.device)
            p_words = p_words.to(args.device)
            p_mask = p_mask.to(args.device)
            h_words = h_words.to(args.device)
            h_mask = h_mask.to(args.device)
            label = label.to(args.device)
            optimizer.zero_grad()
            logits, prob = model(p_words, p_mask, h_words, h_mask)
            loss = criterion(logits, label)

            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()
        acc, dev_loss = evaluate(args, model, val_loader)
        if acc>best_acc:
            best_acc = acc
            bad_count = 0
            torch.save(model.state_dict(), args.save_dict)
        else:
            bad_count += 1
        msg = 'Val Loss: {0:>5.2}, Current acc: {1:>6.2%}, Best acc: {2:>6.2%}, Bad acount: {3}'
        print(msg.format(dev_loss, acc, best_acc, bad_count))
        if bad_count>args.bad_count:
            print('No optimization for a long time, auto-stopping ...')
            break
        # scheduler.step(0.8)
    test(args, model, test_loader)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


