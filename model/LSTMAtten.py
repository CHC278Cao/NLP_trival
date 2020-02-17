import pdb
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class AttentionNet(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nlayers, ntags, dropout, device):
        super(AttentionNet, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm1 = nn.LSTM(emb_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru1 = nn.GRU(hidden_size*2, 64, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(64 * 2, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, ntags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        seq_len = [len(x) for x in input]
        input = pad_sequence(input, batch_first=True)
        emb = self.embedding(input)
        pad_input = pack_padded_sequence(emb, lengths=seq_len, batch_first=True)
        pad_input = pad_input.to(self.device)
