import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', default='atec', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_length', default=50, type=int)

    # model
    parser.add_argument('--model', default='ESIM', type=str)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--vocab_size', default=2000, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    # training
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--save_dict', default='', type=str)
    parser.add_argument('--required_improvement', default=500, type=int)
    parser.add_argument('--bad_count', default=5, type=int)

    args = parser.parse_args()
    args.save_dict = './save_dict/' + args.model

    return args
