from torchcrf import CRF
import torch.nn as nn

class LSTMCRF(nn.Module):
    def __init__(self, args, word2id, tag2id):
        super(LSTMCRF, self).__init__()

        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = len(word2id)+1
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout()

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=1, bidirectional=True, batch_first=False)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first=True)

    def forward(self, x, mask):
        embedding = self.embedding_layer(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        outputs = self.crf.decode(outputs, mask=mask)
        return outputs

    def log_likehood(self, x, tags, mask):
        embedding = self.embedding_layer(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return -self.crf(outputs, tags, mask=mask)
