import torch
import torch.nn as nn
import logging
import numpy as np

from braindecode.torch_ext.util import np_to_var
from torch.autograd import Variable

logger = logging.getLogger(__name__)
hidden_size = 32

class ChronoNet(nn.Module):
    class InceptionBlock(nn.Module):
        def __init__(self, in_size, batch_norm, out_size=32):
            super().__init__()
            self.conv_1 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=(2,1), stride=2, padding=0)
            self.conv_2 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=(4,1), stride=2, padding=1)
            self.conv_3 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=(8,1), stride=2, padding=3)

            self.batch_norm = batch_norm
            if batch_norm:
                self.bnn = nn.BatchNorm1d(num_features=3*out_size)

            self.non_linearity = nn.ReLU

        def forward(self, x):
            # Transpose to  N x C x L
            x = torch.transpose(x, 1, 2)
            x = torch.cat([self.conv_1(x), self.conv_2(x), self.conv_3(x)], dim=1)
            x = self.non_linearity()(x)
            x = torch.transpose(x, 1, 2)
            return x

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inception_block_1 = ChronoNet.InceptionBlock(22, batch_norm=False, out_size=32)
        self.inception_block_2 = ChronoNet.InceptionBlock(32*3, batch_norm=False, out_size=32)
        self.inception_block_3 = ChronoNet.InceptionBlock(32*3, batch_norm=False, out_size=32)

        self.gru_1 = nn.GRU(input_size=32*3, hidden_size=32, num_layers=1)
        self.gru_2 = nn.GRU(input_size=32, hidden_size=32, num_layers=1)
        self.gru_3 = nn.GRU(input_size=32, hidden_size=32, num_layers=1)
        self.gru_4 = nn.GRU(input_size=32, hidden_size=32, num_layers=1)

        self.fc = nn.Linear(32, 4, bias=True)

    def forward(self, x, hidden):
        x = self.inception_block_1(x)
        x = self.inception_block_2(x)
        x = self.inception_block_3(x)

        x, hidden = self.gru_1(x, hidden)
        x, hidden = self.gru_2(x, hidden)
        x, hidden = self.gru_3(x, hidden)
        x, hidden = self.gru_4(x, hidden)
        x = self.fc(x)

        return x, hidden

    def create_network(self):
        # model = nn.Sequential()
        # model.add_module('1', self.inception_block_1)
        # model.add_module('2', self.inception_block_2)
        # model.add_module('3', self.inception_block_3)
        #
        # model.add_module('5', self.gru_1)
        # model.add_module('6', self.gru_2)
        # model.add_module('7', self.gru_3)
        # model.add_module('8', self.gru_4)
        # return model
        return self

    def offset_size(self, sequence_size):
        assert sequence_size % 8 == 0,  "For this model it is better if sequence size is divisible by 8"
        return sequence_size - sequence_size//8

    def init_hidden(self):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, 1, hidden_size).zero_())

model = ChronoNet().create_network()
out = model.forward(np_to_var(np.ones(
    (2, 1125, 22, 1), dtype=np.float32)), Variable(model.init_hidden().data),
    )
pass