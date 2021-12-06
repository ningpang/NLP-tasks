import time
import torch.nn as nn
import torch
from datetime import timedelta

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(args, model, train_loader, word_to_id, id_to_word):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
        model.train()
        for i, words in enumerate(train_loader):
            model.zero_grad()
            model.to(args.device)
            words = words.to(args.device)
            optimizer.zero_grad()

            words = words.transpose(1, 0).contiguous()
            inputs, targets = words[:-1, :], words[1:, :]
            targets = targets.view(-1)
            hidden = model.init_hidden(args.layer_num, inputs.size()[1])
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                print('Epoch: %d, Loss: %f'%(epoch, loss.data))
        torch.save(model.state_dict(), args.save_dict)
        title = '春江花月夜凉如水'
        gen_poetry = ''.join(evaluate(args, model, title, word_to_id, id_to_word))
        print(gen_poetry)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

def evaluate(args, model, title, word_to_id, id_to_word):
    poetry = []
    pre_word = '<START>'

    sentence_count = 0
    if title is not None:
        generate_num = len(title)
    else:
        generate_num = args.generate_num

    # 准备第一步要输入的数据
    input = torch.tensor([word_to_id[pre_word]]).view(1, 1).to(args.device)
    hidden = model.init_hidden(args.layer_num, 1)

    for i in range(args.max_generate):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        word = id_to_word[top_index]

        if title is not None:
            if pre_word in ['。', '！', '<START>']:
                if sentence_count == generate_num:
                    break
                else:
                    word = title[sentence_count]
                    sentence_count += 1
                    input = (input.data.new([word_to_id[word]])).view(1, 1)
            else:
                input = (input.data.new([top_index])).view(1, 1)
        else:
            if word == '<EOP>':
                break
            if word in ['。', '！']:
                sentence_count += 1
                if sentence_count == generate_num:
                    poetry.append(word)
                    break

        poetry.append(word)
        pre_word = word
    return poetry

def test(args, model, title, word_to_id, id_to_word):
    model.load_state_dict(torch.load(args.save_dict))
    model.eval()
    model.to(args.device)
    start_time = time.time()
    print('生成藏头诗: ', title)
    poetry = ''.join(evaluate(args, model, title, word_to_id, id_to_word))
    print(poetry)
    time_dif = get_time_dif(start_time)
    print('Time usage: ', time_dif)
    print('随机生成诗: ')
    poetry = ''.join(evaluate(args, model, None, word_to_id, id_to_word))
    print(poetry)
    time_dif = get_time_dif(start_time)
    print('Time usage: ', time_dif)
