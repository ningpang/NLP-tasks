import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', default='poetry', type=str)
    parser.add_argument('--category', default='poet.tang', type=str)
    parser.add_argument('--author', default=None, type=str)
    parser.add_argument('--constrain', default=None, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_length', default=125, type=int)
    parser.add_argument('--max_generate', default=200, type=int)

    # model
    parser.add_argument('--model', default='Poetry', type=str)
    parser.add_argument('--embedding_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--layer_num', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    # training
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--save_dict', default='', type=str)
    parser.add_argument('--required_improvement', default=500, type=int)
    parser.add_argument('--generate_num', default=8, type=int)

    args = parser.parse_args()
    args.save_dict = './save_dict/' + args.model

    return args
