
import pdb
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class RNNModel(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, nlayers, ntags, dropout, device):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn1 = nn.LSTM(emb_size, hidden_size, nlayers, batch_first=True, bidirectional=True)
        # self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size)
        self.linear1 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, ntags)
        self.device = device

        self.initWeight()

    def forward(self, input):
        # pdb.set_trace()
        seq_len = [len(x) for x in input]
        input = pad_sequence(input, batch_first=True).to(self.device)
        emb = self.embedding(input)
        pad_input = pack_padded_sequence(emb, lengths=seq_len, batch_first=True)
        out, (hidden, cell) = self.rnn1(pad_input)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out_avg = torch.mean(out, dim = 1)
        out_max, _ = torch.max(out, dim = 1)
        out = torch.cat((out_avg, out_max), dim = 1)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out


    def initWeight(self):
        for m in self.modules():
            # pdb.set_trace()
            if isinstance(m, (nn.LSTM, nn.GRU)):
                for param in m.parameters():
                    if len(param.size()) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        torch.nn.init.zeros_(param.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)




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





