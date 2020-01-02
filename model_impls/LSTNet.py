__author__ = "Guan Song Wang"

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTNetModel(nn.Module):
    def __init__(self, num_channels, num_classes, window=24*10, hidRNN=100, hidCNN=100, hidSkip=5, CNN_kernel=6, skip=24, highway_window=24, dropout=0.2, output_fun='sigmoid'):
        super(LSTNetModel, self).__init__()
        self.P = window
        self.m = num_channels
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip

        self.num_classes = num_classes
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if (self.skip > 0):
            self.pt = (self.P - self.Ck) / self.skip
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, num_classes)
        else:
            self.linear1 = nn.Linear(self.hidR, num_classes)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
            self.highway_linear = nn.Linear(num_channels, num_classes)
        self.output = None
        if (output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if (output_fun == 'tanh'):
            self.output = torch.tanh
        if (output_fun == 'softmax'):
            self.output = torch.nn.Softmax()

    def forward(self, x):
        x = x.squeeze(dim=3)
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))



        # skip-rnn

        if (self.skip > 0):
            self.pt=int(self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous()

            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(x.shape[0], -1)
            z = self.highway_linear(z)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res
