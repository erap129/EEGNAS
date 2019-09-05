import torch
import torch.nn as nn
from torch.nn import Parameter
import logging


logger = logging.getLogger(__name__)


class LassoFeatureSelection(nn.Module):
    """
    1 to 1 layer that should be used as the first layer in the network. Strong L1 regularization should enforce
    feature selection behavior. Does nothing if initialized with 0.0 lasso_value
    """
    def __init__(self, input_size, lasso_value=0.0):
        super().__init__()

        self.lasso_value = lasso_value
        if self.lasso_value != 0:
            self.mul = Parameter(torch.ones(input_size))

    def forward(self, x):
        if self.lasso_value != 0:
            return x * self.mul
        else:
            return x

    def loss(self):
        if self.lasso_value != 0:
            return self.mul.norm(1) * self.lasso_value
        else:
            return 0

    def get_values(self):
        if self.lasso_value != 0:
            return self.mul.cpu().data.numpy()
        else:
            return 0
