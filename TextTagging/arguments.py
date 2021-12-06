import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', default='renmin', type=str)
    parser.add_argument('--vocab_size', default=4000, type=int)
    parser.add_argument('--max_length', default=60, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    # model
    parser.add_argument('--model', default='LSTMCRF', type=str)
    parser.add_argument('--embedding_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    # training
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--save_dict', default='', type=str)
    parser.add_argument('--bad_count', default=5, type=int)

    args = parser.parse_args()
    args.save_dict = './save_dict/' + args.model

    return args
