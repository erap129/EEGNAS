import random
from EEGNAS import global_vars


class Layer:
    def __init__(self, name=None):
        self.name = name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    def __str__(self):
        rep = type(self).__name__
        if type(self) == ConvLayer:
            rep += f' ({self.kernel_height},{self.kernel_width})_{self.filter_num}'
        elif type(self) == PoolingLayer:
            rep += f' ({self.pool_height},{self.pool_width})_({self.stride_height},{self.stride_width})'
        return rep


class InputLayer(Layer):
    def __init__(self, shape_height, shape_width):
        Layer.__init__(self)
        self.shape_height = shape_height
        self.shape_width = shape_width


class FlattenLayer(Layer):
    def __init__(self):
        Layer.__init__(self)


class DropoutLayer(Layer):
    def __init__(self, rate=0.5):
        Layer.__init__(self)
        self.rate = rate


class BatchNormLayer(Layer):
    def __init__(self, axis=3, momentum=0.1, epsilon=1e-5):
        Layer.__init__(self)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon


class ActivationLayer(Layer):
    def __init__(self, activation_type='elu'):
        Layer.__init__(self)
        self.activation_type = activation_type


class ConvLayer(Layer):
    # @initializer
    def __init__(self, kernel_height=None, kernel_width=None, filter_num=None, stride=None, dilation_height=None, name=None):
        Layer.__init__(self, name)
        if kernel_height is None:
            kernel_height = random.randint(1, global_vars.get('kernel_height_max'))
        if kernel_width is None:
            kernel_width = random.randint(1, global_vars.get('kernel_width_max'))
        if filter_num is None:
            filter_num = random.randint(1, global_vars.get('filter_num_max'))
        if stride is None:
            stride = random.randint(1, global_vars.get('conv_stride_max'))
        if dilation_height is None:
            dilation_height = random.randint(1, global_vars.get('max_dilation_height'))
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.filter_num = filter_num
        self.stride = stride
        self.dilation_height = dilation_height


class PoolingLayer(Layer):
    # @initializer
    def __init__(self, pool_height=None, pool_width=None, stride_height=None, stride_width=None, mode='max'):
        Layer.__init__(self)
        if pool_height is None:
            pool_height = random.randint(1, global_vars.get('pool_height_max'))
        if pool_width is None:
            pool_width = random.randint(1, global_vars.get('pool_width_max'))
        if stride_height is None:
            stride_height = random.randint(1, global_vars.get('pool_stride_height_max'))
        if stride_width is None:
            stride_width = random.randint(1, global_vars.get('pool_stride_width_max'))
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.mode = mode


class IdentityLayer(Layer):
    def __init__(self):
        Layer.__init__(self)


class ZeroPadLayer(Layer):
    def __init__(self, height_pad_top, height_pad_bottom, width_pad_left, width_pad_right):
        Layer.__init__(self)
        self.height_pad_top = height_pad_top
        self.height_pad_bottom = height_pad_bottom
        self.width_pad_left = width_pad_left
        self.width_pad_right = width_pad_right


class ConcatLayer(Layer):
    def __init__(self, first_layer_index, second_layer_index):
        Layer.__init__(self)
        self.first_layer_index = first_layer_index
        self.second_layer_index = second_layer_index


class AveragingLayer(Layer):
    def __init__(self):
        pass


class LayerBlock(Layer):
    def __init__(self, length):
        layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
        Layer.__init__(self)
        self.layers = [layers[random.randint(0, 5)]() for i in range(length)]


