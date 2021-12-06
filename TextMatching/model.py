import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalDropout(nn.Dropout):
    def forward(self, input_tensor):
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = F.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1)*input_tensor

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, vector_size, drop=0.5):
        super(EmbeddingLayer, self).__init__()
        self.vector_size = vector_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, vector_size)
        self.drop = VariationalDropout(drop)

    def load(self, vector):
        self.embedding.weight.data.copy_(vector)

    def forward(self, x):
        return self.drop(self.embedding(x))

class Encoderlayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoderlayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return output

class LocalInferenceModel(nn.Module):
    def __init__(self):
        super(LocalInferenceModel, self).__init__()
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, p, h, p_mask, h_mask):
        e = torch.matmul(p, h.transpose(1, 2))

        inference_mask = torch.matmul(p_mask.unsqueeze(2).float(),
                                      h_mask.unsqueeze(1).float())
        e.masked_fill_(inference_mask<1e-7, -1e7)

        p_score, h_score = self.softmax1(e), self.softmax2(e)

        h_ = h_score.transpose(1, 2).bmm(p)
        p_ = p_score.bmm(h)

        m_p = torch.cat((p, p_, torch.abs(p-p_), p*p_), dim=-1)
        m_h = torch.cat((h, h_, torch.abs(h-h_), h*h_), dim=-1)

        assert inference_mask.shape == e.shape
        assert p.shape == p_.shape and h.shape == h_.shape
        assert m_p.shape[-1] == p.shape[-1]*4

        return m_p, m_h

class CompositionLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.5):
        super(CompositionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, output_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=1, bidirectional=True)
        self.drop = VariationalDropout(dropout)

    def forward(self, m):
        y = self.drop(self.fc(m))
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(y)
        return outputs

class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

    def forward(self, x, x_mask):

        mask_expand = x_mask.unsqueeze(-1).expand(x.shape)


        # avg pooling
        x_ = x*mask_expand
        # v_avg = torch.div(x_.sum(1), x_mask.sum(1).unsqueeze(-1))
        v_avg = x_.mean(1)

        # max pooling
        x_ = x.masked_fill(mask_expand==0, -1e7)
        v_max = x_.max(1).values

        return torch.cat((v_avg, v_max), dim=-1)
        # return v_avg


class InferenceComposition(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.2):
        super(InferenceComposition, self).__init__()
        self.composition = CompositionLayer(input_size, output_size, hidden_size, dropout)
        self.pooling = Pooling()

    def forward(self, m_p, m_h, p_mask, h_mask):

        v_p, v_h = self.composition(m_p), self.composition(m_h)
        v_p_, v_h_ = self.pooling(v_p, p_mask), self.pooling(v_h, h_mask)
        v = torch.cat((v_p_, v_h_), dim=-1)
        return v

class LinearSoftmax(nn.Module):
    def __init__(self, input_size, output_size, class_num, activation='relu', dropout=0.5):
        super(LinearSoftmax, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Unknown activation function ...')
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            self.drop,
            nn.Linear(input_size, output_size),
            self.activation,
            nn.Linear(output_size, class_num)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits

class ESIM(nn.Module):
    def __init__(self, args):
        super(ESIM, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.dropout_rate = args.dropout
        self.vocab_size = args.vocab_size
        self.embedding = EmbeddingLayer(self.vocab_size, self.embedding_dim, self.dropout_rate)
        self.encoder = Encoderlayer(self.embedding_dim, self.hidden_dim)
        self.inference = LocalInferenceModel()
        self.infercomp = InferenceComposition(self.hidden_dim*8, self.hidden_dim, self.hidden_dim, self.dropout_rate)
        self.linear = LinearSoftmax(self.hidden_dim*8, self.hidden_dim, 2, activation='tanh')

    def forward(self, p, p_mask, h, h_mask):
        p_embed = self.embedding(p)
        h_embed = self.embedding(h)

        p_encoding = self.encoder(p_embed)
        h_encoding = self.encoder(h_embed)

        m_p, m_h = self.inference(p_encoding, h_encoding, p_mask, h_mask)


        v = self.infercomp(m_p, m_h, p_mask, h_mask)

        logits = self.linear(v)
        prob = F.softmax(logits, dim=-1)

        return logits, prob
