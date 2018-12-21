from torch import nn
from braindecode.torch_ext.util import np_to_var
from braindecode.models.util import to_dense_prediction_model
import numpy as np

model = nn.Sequential()
conv1 = nn.Conv2d(1, 1, (2, 1), stride=1)
conv1.weight.data.fill_(1)
conv1.bias.data.fill_(0)
conv2 = nn.Conv2d(1, 1, (2, 1), stride=1)
conv2.weight.data.fill_(1)
conv2.bias.data.fill_(0)
model.add_module("conv1", conv1)
model.add_module("pool1", nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
model.add_module("conv2", conv2)
model.add_module("pool2", nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

input = np_to_var(np.arange(0, 13, 1)[np.newaxis, np.newaxis, :, np.newaxis])
input = input.type('torch.FloatTensor')

input1 = np_to_var(np.arange(0, 10, 1)[np.newaxis, np.newaxis, :, np.newaxis])
input1 = input1.type('torch.FloatTensor')

input2 = np_to_var(np.arange(1, 11, 1)[np.newaxis, np.newaxis, :, np.newaxis])
input2 = input2.type('torch.FloatTensor')

input3 = np_to_var(np.arange(2, 12, 1)[np.newaxis, np.newaxis, :, np.newaxis])
input3 = input3.type('torch.FloatTensor')

input4 = np_to_var(np.arange(3, 13, 1)[np.newaxis, np.newaxis, :, np.newaxis])
input4 = input4.type('torch.FloatTensor')

print('output 1:' + str(model.forward(input1)))
print('output 2:' + str(model.forward(input2)))
print('output 3:' + str(model.forward(input3)))
print('output 4:' + str(model.forward(input4)))
out_regular = model.forward(input)

to_dense_prediction_model(model)
print('total output:' + str(model.forward(input)))


pass