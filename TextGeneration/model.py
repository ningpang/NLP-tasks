import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PoetryModel(nn.Module):
    def __init__(self, args, vocab_size, layer_num):
        super(PoetryModel, self).__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=layer_num)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs, hidden):
        seq_len, batch_size = inputs.size()
        embedding = self.embedding(inputs)
        outputs, hidden = self.lstm(embedding, hidden)

        output = F.relu(self.fc(outputs.view(seq_len*batch_size, -1)))
        return output, hidden

    def init_hidden(self, layer_num, batch_size):
        return (Variable(torch.zeros(layer_num, batch_size, self.hidden_dim).to(self.args.device)),
                Variable(torch.zeros(layer_num, batch_size, self.hidden_dim).to(self.args.device)))
