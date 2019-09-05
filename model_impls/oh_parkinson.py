from braindecode.torch_ext.util import np_to_var
import numpy as np
from braindecode.torch_ext.modules import Expression
from torch import nn
from EEGNAS import global_vars


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


class OhParkinson:
    @staticmethod
    def create_network():
        model = nn.Sequential()
        model.add_module('conv_1', nn.Conv2d(global_vars.get('eeg_chans'), 5,
                                             kernel_size=(20, 1), stride=1))
        model.add_module('pool_1', nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        model.add_module('conv_2', nn.Conv2d(5, 10, kernel_size=(10, 1), stride=1))
        model.add_module('pool_2', nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        model.add_module('conv_3', nn.Conv2d(10, 10, kernel_size=(10, 1), stride=1))
        model.add_module('pool_3', nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        model.add_module('conv_4', nn.Conv2d(10, 15, kernel_size=(5, 1), stride=1))
        model.add_module('pool_4', nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        model.add_module('flatten', Flatten())

        input_shape = (2, global_vars.get('eeg_chans'), global_vars.get('input_time_len'), 1)
        out = model.forward(np_to_var(np.ones(input_shape, dtype=np.float32)))
        dim = 1
        for muldim in out.shape[1:]:
            dim *= muldim
        model.add_module('dense_1', nn.Linear(in_features=dim, out_features=20))
        model.add_module('dropout_1', nn.Dropout(p=0.5))
        model.add_module('dense_2', nn.Linear(in_features=20, out_features=10))
        model.add_module('dropout_2', nn.Dropout(p=0.5))
        model.add_module('dense_3', nn.Linear(in_features=10, out_features=global_vars.get('n_classes')))
        model.add_module('softmax', nn.Softmax())
        return model
