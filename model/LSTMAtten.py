import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class AttentionNet(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nlayers, ntags, dropout, device):
        super(AttentionNet, self).__init__()

        self.device = device

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm1 = nn.LSTM(emb_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(4 * hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, ntags)
        self.dropout = nn.Dropout(dropout)


    def attention_net(self, enc_output, enc_hidden, mask = None):
        """
            get the attention output
        :param enc_output: encoder output: [batch_size, seq_len, hidden * 2]
        :param enc_hidden: encoder hidden: [2, batch_size, hidden]
        :param mask: mask for calculating attention weight
        :return: attention weight
        """
        # pdb.set_trace()
        hidden_state = torch.cat((enc_hidden[0], enc_hidden[1]), dim = 1).unsqueeze(dim = 2)
        e_t = torch.bmm(enc_output, hidden_state).squeeze(dim = 2)

        if mask is not None:
            e_t.data.masked_fill_(mask.bool(), -float("inf"))

        alpha_t = F.softmax(e_t, dim = 1)
        alpha_t = alpha_t.unsqueeze(dim = 1) # [batch_size, 1, seq_len]
        a_t = torch.bmm(alpha_t, enc_output).squeeze(dim = 1) # [batch_size, 2 * hidden]
        U_t = torch.cat((a_t, hidden_state.squeeze(dim = 2)), dim = 1) # [batch_size, 4 * hidden]

        return U_t

    def generate_mask(self, enc_output, seq_len):
        # pdb.set_trace()
        enc_mask = torch.zeros(enc_output.size(0), enc_output.size(1), dtype = torch.float)
        for line_id, line_len in enumerate(seq_len):
            enc_mask[line_id, line_len: ] = 1
        return enc_mask.to(self.device)


    def forward(self, input):
        # pdb.set_trace()
        seq_len = [len(x) for x in input]
        input = pad_sequence(input, batch_first=True).to(self.device)
        emb = self.embedding(input)
        pad_input = pack_padded_sequence(emb, lengths=seq_len, batch_first=True)
        enc_out, (enc_hidden, enc_cell) = self.lstm1(pad_input)
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        mask = self.generate_mask(enc_out, seq_len)
        attn_output = self.attention_net(enc_out, enc_hidden, mask)
        out = self.linear1(attn_output)
        out = self.dropout(self.relu(out))
        out = self.linear2(out)

        return out


def train_epoch(model, dataloader, device, optimizer, criterion):
    model.train()
    model.to(device)
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        # pdb.set_trace()
        target = torch.stack(target).view(-1).to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (batch_idx % 50) == 49:
            print("Batch: {}, Loss: {}".format(batch_idx + 1, loss.item()))

    return running_loss / len(dataloader)

def valid_epoch(model, validloader, device, criterion):
    model.eval()
    model.to(device)
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validloader):
            out = model(data)
            target = torch.stack(target).view(-1).to(device)
            loss = criterion(out, target)
            pred = torch.argmax(out, dim = 1)
            rect = (torch.eq(pred.detach(), target)).cpu().sum()
            correct += rect.item()
            running_loss += loss.item()

    return running_loss / len(validloader), correct / len(validloader)


