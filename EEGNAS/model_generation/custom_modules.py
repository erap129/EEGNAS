import torch

from torch import nn
from torch.nn import init

from EEGNAS import global_vars


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


class AveragingModule(nn.Module):
    def forward(self, inputs):
        return torch.mean(inputs, dim=0)


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_neurons, n_networks, true_avg=False):
        super(LinearWeightedAvg, self).__init__()
        self.weight_inputs = []
        for network_idx in range(n_networks):
            self.weight_inputs.append(nn.Parameter(torch.randn(1, n_neurons)))
            init.xavier_uniform_(self.weight_inputs[-1], gain=1)
        if true_avg:
            for network_idx in range(n_networks):
                self.weight_inputs[network_idx].data = torch.tensor([[1/n_networks for i in range(n_neurons)]]).view((1, n_neurons))
        self.weight_inputs = nn.ParameterList(self.weight_inputs)

    def forward(self, *inputs):
        res = 0
        for inp_idx, input in enumerate(inputs):
            res += input * self.weight_inputs[inp_idx]
        return res


class BasicEnsemble(nn.Module):
    def __init__(self, networks, out_size):
        super(BasicEnsemble, self).__init__()
        self.networks = networks
        self.linear = torch.nn.Linear(out_size * len(networks), out_size)

    def forward(self, X):
        concat_out = torch.concat([model(X) for model in self.networks])
        res = self.linear(concat_out)
        return res


class _squeeze_final_output(nn.Module):
    def __init__(self):
        super(_squeeze_final_output, self).__init__()

    def forward(self, x):
        if x.size()[3] == 1:
            x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
        return x


class _transpose(nn.Module):
    def __init__(self, shape):
        super(_transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)


class AveragingEnsemble(nn.Module):
    def __init__(self, models, true_avg=True):
        super(AveragingEnsemble, self).__init__()
        self.avg_layer = LinearWeightedAvg(global_vars.get('n_classes'), len(models), true_avg)
        self.models = models
        self.softmax = nn.Softmax()
        self.flatten = _squeeze_final_output()

    def set_model_weights(self, freeze):
        for model in self.models:
            for child in model.children():
                for param in child.parameters():
                    param.requires_grad = freeze

    def forward(self, input):
        outputs = []
        for model in self.models:
            outputs.append(model(input))
        avg_output = self.avg_layer(*outputs)
        if global_vars.get('problem') == 'classification':
            avg_output = self.softmax(avg_output)
        return avg_output
