import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', default='cnews', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--max_length', default=600, type=int)
    parser.add_argument('--num_class', default=10, type=int)

    # model
    parser.add_argument('--model', default='cnn', type=str)
    parser.add_argument('--vocab_size', default=5000, type=int)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256)
    parser.add_argument('--dropout', default=0.8, type=float)

    #training
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--save_dict', default='', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--required_improvement', default=1000, type=int)

    args = parser.parse_args()
    args.save_dict = './save_dict/'+ args.model

    return args
