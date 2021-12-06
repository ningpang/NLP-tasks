import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sklearn import metrics

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(args, model, train_loader, val_loader, test_loader, class_list):
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    model.train()

    for epoch in range(args.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
        for i, (tokens, labels, length, mask) in enumerate(train_loader):
            model.zero_grad()
            model.to(args.device)
            tokens = tokens.to(args.device)
            labels = labels.to(args.device)
            outputs = model(tokens)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if total_batch%100 == 0:
                dev_acc, dev_loss = evaluate(args, model, val_loader, class_list)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), args.save_dict)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Val Loss: {1:>5.2}, Time: {2} {3}'
                print(msg.format(total_batch, dev_loss, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch-last_improve>args.required_improvement:
                print('No optimization for a long time, auto-stopping ...')
                flag = True
                break

        if flag:
            break
    test(args, model, test_loader, class_list)

def evaluate(args, model, data_loader, class_list, test=False):
    model.eval()
    device = args.device
    model.to(device)
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for (tokens, labels, seq_len, mask) in data_loader:
            tokens = tokens.to(args.device)
            labels = labels.to(args.device)
            outputs = model(tokens)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        reprot = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total/len(data_loader), reprot, confusion

    return acc, loss_total/len(data_loader)

def test(args, model, data_loader, class_list):
    model.load_state_dict(torch.load(args.save_dict))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_reprot, test_confusion = evaluate(args, model, data_loader, class_list, test=True)
    msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall, and F1-Score... ")
    print(test_reprot)
    print("Confusion Matrix")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)