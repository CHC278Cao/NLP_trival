import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_size, emb_size, dropout, ntags):
        super(CNNModel, self).__init__()
        self.embeddimg = nn.Embedding(input_size, emb_size)
        filter_size = [1, 2, 3, 5]
        layers = [nn.Conv1d(in_channels=emb_size, out_channels=128, kernel_size=k, bias=True) for k in filter_size]
        self.conv1 = nn.ModuleList(layers)

        # self.conv1 = nn.Sequential([
        #     nn.Conv1d(in_channels=emb_size, out_channels=50, )
        # ])

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(filter_size) * 128, ntags)

    def forward(self, input):
        # pdb.set_trace()
        emb = self.embeddimg(input)
        emb = emb.permute(0, 2, 1)
        out = [F.relu(conv(emb)) for conv in self.conv1]
        out = [F.max_pool1d(x, x.size(2)).squeeze(dim = 2) for x in out]
        out = torch.cat(out, dim = 1)
        out = self.dropout(out)
        logit = self.linear(out)

        return logit


def train_epoch(model, dataloader, device, optimizer, criterion):
    model.train()
    model.to(device)
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        # pdb.set_trace()
        data = torch.stack(data).to(device)
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
            # pdb.set_trace()
            data = torch.stack(data).to(device)
            target = torch.stack(target).view(-1).to(device)
            out = model(data)
            loss = criterion(out, target)
            pred = torch.argmax(out, dim = 1)
            rect = (torch.eq(pred.detach(), target)).cpu().sum()
            correct += rect.item()
            running_loss += loss.item()

    return running_loss / len(validloader), correct / len(validloader)
