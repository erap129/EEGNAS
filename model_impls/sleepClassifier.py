import torch
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var

from EEGNAS import global_vars
from torch import nn
from torchsummary import summary
import numpy as np
C=30

class MyModel:
    # remove empty dim at end and potentially remove empty time dim
    # do not just use squeeze as we never want to remove first dim
    @staticmethod
    def _squeeze_final_output(x):
        assert x.size()[3] == 1
        x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
        return x

    @staticmethod
    def _transpose_time_to_spat(x):
        return x.permute(0, 3, 2, 1)

    @staticmethod
    def _transpose_shift_and_swap(x):
        return x.permute(0, 3, 1, 2)

    @staticmethod
    def _transpose_channels_with_length(x):
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _stack_input_by_time(x):
        if global_vars.config['DEFAULT']['channel_dim'] == 'one':
            return x.view(x.shape[0], -1, int(x.shape[2] / global_vars.get('time_factor')), x.shape[3])
        else:
            return x.view(x.shape[0], x.shape[1], int(x.shape[2] / global_vars.get('time_factor')), -1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_sleep_classifier():
    model = nn.Sequential()

    model.add_module('permute_1', Expression(MyModel._transpose_shift_and_swap))
    model.add_module('conv_1', nn.Conv2d(1, global_vars.get('eeg_chans'),
                                         kernel_size=(global_vars.get('eeg_chans'), 1)))
    model.add_module('permute_2', Expression(MyModel._transpose_channels_with_length))
    model.add_module('conv_2', nn.Conv2d(1, 8, kernel_size=(1, 64), stride=1))
    model.add_module('pool_1', nn.MaxPool2d(kernel_size=(1, 16), stride=(1, 1)))
    model.add_module('conv_3', nn.Conv2d(8, 8, kernel_size=(1, 64), stride=1))
    model.add_module('pool_2', nn.MaxPool2d(kernel_size=(1, 16), stride=(1, 1)))
    model.add_module('flatten', Flatten())
    model.add_module('dropout', nn.Dropout(p=0.5))

    input_shape = (2, global_vars.get('eeg_chans'), global_vars.get('input_time_len'), 1)
    out = model.forward(np_to_var(np.ones(input_shape, dtype=np.float32)))
    dim = 1
    for muldim in out.shape[1:]:
        dim *= muldim
    model.add_module('dense', nn.Linear(in_features=dim, out_features=global_vars.get('n_classes')))
    model.add_module('softmax', nn.Softmax())

    return model

    #
    # summary(model, (globals.get('eeg_chans'), 1125, 1))
    #
    # pass

