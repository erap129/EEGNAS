import torch
from copy import deepcopy

from braindecode.torch_ext.modules import Expression
from torch import nn
from model_generation.custom_modules import IdentityModule, _squeeze_final_output


def handle_deconv(conv_layer):
    in_channels = conv_layer.out_channels
    out_channels = conv_layer.in_channels
    kernel_size = conv_layer.kernel_size
    deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
    deconv_layer.weight.data = conv_layer.weight.data
    return deconv_layer


def handle_depool(pooling_layer):
    depool_layer = nn.MaxUnpool2d(kernel_size=pooling_layer.kernel_size, stride=pooling_layer.stride)
    return depool_layer


def handle_other(layer):
    return layer


def handle_squeeze(layer):
    return _unsqueeze_final_output()


class _unsqueeze_final_output(nn.Module):
    def __init__(self):
        super(_unsqueeze_final_output, self).__init__()

    def forward(self, x):
        if x.ndims == 3:
            x = x[:, :, :, None]
        elif x.ndims == 2:
            x = x[:, :, None, None]
        return x


handle_layer = {nn.Conv2d: handle_deconv,
                nn.MaxPool2d: handle_depool,
                nn.BatchNorm2d: handle_other,
                IdentityModule: handle_other,
                nn.Dropout: handle_other,
                nn.ELU: handle_other,
                _squeeze_final_output: handle_squeeze,
                Expression: handle_squeeze
                }


class DeconvNet(nn.Module):
    def __init__(self, model):
        super(DeconvNet, self).__init__()
        self.conv2deconv = {}
        self.deconv_nn = nn.Sequential()
        for idx, layer in reversed(list(enumerate(model.children()))):
            if type(layer) not in [nn.Softmax]:
                self.deconv_nn.add_module(f'{type(layer).__name__}_{idx}', handle_layer[type(layer)](layer))
                if type(layer) == nn.MaxPool2d:
                    self.conv2deconv[len(model) - 2 - idx] = idx
        self.deconv_nn.cuda()

    def forward(self, x, start_idx, pool_locs, pool_sizes):
        start_idx = len(self.deconv_nn) - start_idx - 1
        for idx in range(start_idx, len(self.deconv_nn)):
            if isinstance(self.deconv_nn[idx], nn.MaxUnpool2d):
                x = self.deconv_nn[idx](x, pool_locs[self.conv2deconv[idx]], output_size=pool_sizes[self.conv2deconv[idx]])
            else:
                x = self.deconv_nn[idx](x)
        return x


def return_indices_for_maxpool(model):
    for idx, layer in enumerate(model.children()):
        if type(layer) == nn.MaxPool2d:
            model[idx] = nn.MaxPool2d(layer.kernel_size, layer.stride, return_indices=True)


def erase_all_filters_but_one(x, filter_idx):
    filter_data = x[:, filter_idx]
    x = torch.zeros(x.shape).cuda()
    x[:, filter_idx] = filter_data
    return x


class ConvDeconvNet(nn.Module):
    def __init__(self, model):
        super(ConvDeconvNet, self).__init__()
        self.deconv_net = DeconvNet(model)
        self.model = deepcopy(model)
        return_indices_for_maxpool(self.model)
        self.pool_locs = {}
        self.pool_sizes = {}

    def forward(self, x, layer_idx, filter_idx):
        for idx, layer in enumerate(list(self.model.children())[:layer_idx+1]):
            if isinstance(layer, nn.MaxPool2d):
                self.pool_sizes[idx] = x.shape
                x, self.pool_locs[idx] = layer(x)
            else:
                x = layer(x)
        x = erase_all_filters_but_one(x, filter_idx)
        reconstruction = self.deconv_net.forward(x, layer_idx, self.pool_locs, self.pool_sizes)
        return reconstruction



