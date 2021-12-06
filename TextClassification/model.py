import torch
from torch import nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.dropout_rate = args.dropout
        self.num_class = args.num_class
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.max_length = args.max_length
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool1d(self.max_length))
        self.fc = nn.Linear(self.hidden_dim, self.num_class)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.view(-1, x.size(1))
        x = F.dropout(x, self.dropout_rate)
        x = self.fc(x)
        return x


class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_size
        self.hidden_dim = args.hidden_dim
        self.dropout_rate = args.dropout
        self.num_class = args.num_class
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim//2, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, self.num_class),
                                nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x, (h_n, c_n) = self.rnn(x)
        feature = F.dropout(h_n, self.dropout_rate)
        feature_map = torch.cat([feature[i, :, :] for i in range(feature.shape[0])], dim=1)
        logits = self.fc(feature_map)
        return logits
