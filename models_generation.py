from keras.models import Model
from keras.layers import Conv2D, Flatten, Activation, Lambda, Dropout, Input, Cropping2D, Concatenate, ZeroPadding2D, BatchNormalization
import numpy as np
from toposort import toposort_flatten
from keras_models import dilation_pool, mean_layer
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
import copy
WARNING = '\033[93m'
ENDC = '\033[0m'

def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure.values():
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
    print('-----------------------------------------------')


def create_topo_layers(layers):
    layer_dict = {}
    for layer in layers:
        layer_dict[layer.id] = {x.id for x in layer.connections}
    return list(reversed(toposort_flatten(layer_dict)))


class Layer:
    running_id = 0

    def __init__(self, name=None):
        self.id = Layer.running_id
        self.connections = []
        self.parent = None
        self.keras_layer = None
        self.name = name
        Layer.running_id += 1

    def make_connection(self, other):
        self.connections.append(other)
        other.parent = self


class LambdaLayer(Layer):
    def __init__(self, function):
        Layer.__init__(self)
        self.function = function

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
    def __init__(self, kernel_width, kernel_height, filter_num, name=None):
        Layer.__init__(self, name)
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.filter_num = filter_num


class PoolingLayer(Layer):
    def __init__(self, pool_width, stride_width, mode):
        Layer.__init__(self)
        self.pool_width = pool_width
        self.stride_width = stride_width
        self.mode = mode


class CroppingLayer(Layer):
    def __init__(self, height_crop_top, height_crop_bottom, width_crop_left, width_crop_right):
        Layer.__init__(self)
        self.height_crop_top = height_crop_top
        self.height_crop_bottom = height_crop_bottom
        self.width_crop_left = width_crop_left
        self.width_crop_right = width_crop_right


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


def random_model(n_chans, input_time_len):
    layer_collection = {}

    # random variables
    filter_time_length = random.randint(1, 50)
    n_filters_time = random.randint(1, 25)
    n_filters_spat = random.randint(1, 25)
    batchnorm_prob = random.random()
    activation_prob = random.random()
    pool_prob = random.random()

    inputs = InputLayer(shape_height=n_chans, shape_width=input_time_len)
    layer_collection[inputs.id] = inputs
    conv_time = ConvLayer(kernel_width=filter_time_length, kernel_height=1, filter_num=n_filters_time)
    layer_collection[conv_time.id] = conv_time
    inputs.make_connection(conv_time)
    conv_spat = ConvLayer(kernel_width=1, kernel_height=n_chans, filter_num=n_filters_spat)
    layer_collection[conv_spat.id] = conv_spat
    conv_time.make_connection(conv_spat)

    if random.random() > batchnorm_prob:
        batchnorm = BatchNormLayer()
        layer_collection[batchnorm.id] = batchnorm
        conv_spat.make_connection(batchnorm)
    else:
        batchnorm = conv_spat

    if random.random() > activation_prob:
        elu = ActivationLayer()
        layer_collection[elu.id] = elu
        batchnorm.make_connection(elu)
    else:
        elu = batchnorm

    if random.random() > pool_prob:
        maxpool = PoolingLayer(pool_width=3, stride_width=3, mode='MAX')
        layer_collection[maxpool.id] = maxpool
        elu.make_connection(maxpool)

    return MyModel.new_model_from_structure(layer_collection=layer_collection, name='base')


def base_model(n_chans=22, input_time_len=1125, n_filters_time=25, n_filters_spat=25, filter_time_length=10):
    layer_collection = {}
    inputs = InputLayer(shape_height=n_chans, shape_width=input_time_len)
    layer_collection[inputs.id] = inputs
    conv_time = ConvLayer(kernel_width=filter_time_length, kernel_height=1, filter_num=n_filters_time)
    layer_collection[conv_time.id] = conv_time
    inputs.make_connection(conv_time)
    conv_spat = ConvLayer(kernel_width=1, kernel_height=n_chans, filter_num=n_filters_spat)
    layer_collection[conv_spat.id] = conv_spat
    conv_time.make_connection(conv_spat)
    batchnorm = BatchNormLayer()
    layer_collection[batchnorm.id] = batchnorm
    conv_spat.make_connection(batchnorm)
    elu = ActivationLayer()
    layer_collection[elu.id] = elu
    batchnorm.make_connection(elu)
    maxpool = PoolingLayer(pool_width=3, stride_width=3, mode='MAX')
    layer_collection[maxpool.id] = maxpool
    elu.make_connection(maxpool)
    return MyModel(model=None, layer_collection=layer_collection, name='base')


def target_model():
    basemodel = base_model()
    model = add_conv_maxpool_block(basemodel, conv_filter_num=50, conv_layer_name='conv1', random_values=False)
    model = add_conv_maxpool_block(model, conv_filter_num=100, conv_layer_name='conv2', random_values=False)
    model = add_conv_maxpool_block(model, conv_filter_num=200, dropout=True, conv_layer_name='conv3', random_values=False)
    return model


def genetic_filter_experiment_model():
    basemodel = base_model()
    add_conv_layer(base_model, filters=random.randint(1,1000), kernel_width=10, kernel_height=1, in_place=True)
    model = add_conv_layer()



def finalize_model(layer_collection, naive_nas):
    for layer in layer_collection.values():
        layer.keras_layer = None
    layer_collection = copy.deepcopy(layer_collection)
    topo_layers = create_topo_layers(layer_collection.values())
    last_layer_id = topo_layers[-1]
    if naive_nas.cropping:
        final_conv_width = 2
    else:
        final_conv_width = 'down_to_one'
    conv_layer = ConvLayer(kernel_width=final_conv_width, kernel_height=1, filter_num=naive_nas.n_classes)
    layer_collection[conv_layer.id] = conv_layer
    layer_collection[last_layer_id].make_connection(conv_layer)
    softmax = ActivationLayer('softmax')
    layer_collection[softmax.id] = softmax
    conv_layer.make_connection(softmax)
    if naive_nas.cropping:
        mean = LambdaLayer(mean_layer)
        layer_collection[mean.id] = mean
        softmax.make_connection(mean)
    flatten = FlattenLayer()
    layer_collection[flatten.id] = flatten
    if naive_nas.cropping:
        mean.make_connection(flatten)
    else:
        softmax.make_connection(flatten)
    return MyModel.new_model_from_structure(layer_collection)


def add_conv_maxpool_block(model, conv_width=10, conv_filter_num=50, dropout=False,
                           pool_width=3, pool_stride=3, conv_layer_name=None, random_values=True):
    for layer in model.layer_collection.values():
        layer.keras_layer = None
    layer_collection = copy.deepcopy(model.layer_collection)
    if random_values:
        conv_width = random.randint(5, 10)
        conv_filter_num = random.randint(0, 50)
        pool_width = 2
        pool_stride = 2

    topo_layers = create_topo_layers(layer_collection.values())
    last_layer_id = topo_layers[-1]
    if dropout:
        dropout = DropoutLayer()
        layer_collection[dropout.id] = dropout
        layer_collection[last_layer_id].make_connection(dropout)
        last_layer_id = dropout.id
    conv_layer = ConvLayer(kernel_width=conv_width, kernel_height=1,
                           filter_num=conv_filter_num, name=conv_layer_name)
    layer_collection[conv_layer.id] = conv_layer
    layer_collection[last_layer_id].make_connection(conv_layer)
    batchnorm_layer = BatchNormLayer()
    layer_collection[batchnorm_layer.id] = batchnorm_layer
    conv_layer.make_connection(batchnorm_layer)
    activation_layer = ActivationLayer()
    layer_collection[activation_layer.id] = activation_layer
    batchnorm_layer.make_connection(activation_layer)
    maxpool_layer = PoolingLayer(pool_width=pool_width, stride_width=pool_stride, mode='MAX')
    layer_collection[maxpool_layer.id] = maxpool_layer
    activation_layer.make_connection(maxpool_layer)
    return MyModel.new_model_from_structure(layer_collection, name=model.name + '->add_conv_maxpool')

def add_skip_connection_concat(model):
    topo_layers = create_topo_layers(model.layer_collection.values())
    to_concat = random.sample(model.layer_collection.keys(), 2)  # choose 2 random layer id's
    # first_layer_index = topo_layers.index(np.min(to_concat))
    # second_layer_index = topo_layers.index(np.max(to_concat))
    # first_layer_index = topo_layers[first_layer_index]
    # second_layer_index = topo_layers[second_layer_index]
    first_layer_index = np.min(to_concat)
    second_layer_index = np.max(to_concat)
    first_shape = model.model.get_layer(str(first_layer_index)).output.shape
    second_shape = model.model.get_layer(str(second_layer_index)).output.shape
    print('first layer shape is:', first_shape)
    print('second layer shape is:', second_shape)

    height_diff = int(first_shape[1]) - int(second_shape[1])
    width_diff = int(first_shape[2]) - int(second_shape[2])
    height_crop_top = height_crop_bottom = np.abs(int(height_diff / 2))
    width_crop_left = width_crop_right = np.abs(int(width_diff / 2))
    if height_diff % 2 == 1:
        height_crop_top += 1
    if width_diff % 2 == 1:
        width_crop_left += 1
    if height_diff < 0:
        ChosenHeightClass = ZeroPadLayer
    else:
        ChosenHeightClass = CroppingLayer
    if width_diff < 0:
        ChosenWidthClass = ZeroPadLayer
    else:
        ChosenWidthClass = CroppingLayer
    first_layer = model.layer_collection[first_layer_index]
    second_layer = model.layer_collection[second_layer_index]
    next_layer = first_layer
    if height_diff != 0:
        heightChanger = ChosenHeightClass(height_crop_top, height_crop_bottom, 0, 0)
        model.layer_collection[heightChanger.id] = heightChanger
        first_layer.make_connection(heightChanger)
        next_layer = heightChanger
    if width_diff != 0:
        widthChanger = ChosenWidthClass(0, 0, width_crop_left, width_crop_right)
        model.layer_collection[widthChanger.id] = widthChanger
        next_layer.make_connection(widthChanger)
        next_layer = widthChanger
    concat = ConcatLayer(next_layer.id, second_layer_index)
    model.layer_collection[concat.id] = concat
    next_layer.connections.append(concat)
    for lay in second_layer.connections:
        concat.connections.append(lay)
        if not isinstance(lay, ConcatLayer):
            lay.parent = concat
        else:
            if lay.second_layer_index == second_layer_index:
                lay.second_layer_index = concat.id
            if lay.first_layer_index == second_layer_index:
                lay.first_layer_index = concat.id
    second_layer.connections = []
    second_layer.connections.append(concat)
    return model

def edit_conv_layer(model, mode):
    layer_collection = copy.deepcopy(model.layer_collection)
    conv_indices = [layer.id for layer in layer_collection.values() if isinstance(layer, ConvLayer)]
    try:
        random_conv_index = random.randint(2, len(conv_indices) - 2) # don't include last conv
    except ValueError:
        return model
    factor = 1 + random.uniform(0,1)
    if mode == 'filters':
        layer_collection[conv_indices[random_conv_index]].filter_num = \
            int(layer_collection[conv_indices[random_conv_index]].filter_num * factor)
    elif mode == 'kernels':
        layer_collection[conv_indices[random_conv_index]].kernel_width = \
            int(layer_collection[conv_indices[random_conv_index]].kernel_width * factor)
    return MyModel.new_model_from_structure(layer_collection, name=model.name + '->factor_filters')

def factor_filters(model):
    return edit_conv_layer(model, mode='filters')

def factor_kernels(model):
    return edit_conv_layer(model, mode='kernels')

def set_target_model_filters(model, filt1, filt2, filt3):
    conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
    conv_indices = conv_indices[2:len(conv_indices)]  # take only relevant indices
    model.layer_collection[conv_indices[0]].filter_num = filt1
    model.layer_collection[conv_indices[1]].filter_num = filt2
    model.layer_collection[conv_indices[2]].filter_num = filt3
    return model.new_model_from_structure(model.layer_collection)

def set_target_model_kernel_sizes(model, size1, size2, size3):
    conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
    conv_indices = conv_indices[2:len(conv_indices)]  # take only relevant indices
    model.layer_collection[conv_indices[0]].kernel_width = size1
    model.layer_collection[conv_indices[1]].kernel_width = size2
    model.layer_collection[conv_indices[2]].kernel_width = size3
    return model.new_model_from_structure(model.layer_collection)

def mutate_net(model):
    operations = [add_conv_maxpool_block, factor_filters, factor_kernels, add_skip_connection_concat]
    op_index = random.randint(0, len(operations) - 1)
    try:
        model = operations[op_index](model)
    except ValueError as e:
        print('excpetion occured while performing ', operations[op_index].__name__)
        print('error message: ', str(e))
        print('trying another mutation...')
        return mutate_net(model)
    return model

class MyModel:
    def __init__(self, model, layer_collection={}, name=None):
        self.layer_collection = layer_collection
        self.model = model
        self.name = name

    @staticmethod
    def new_model_from_structure(layer_collection, name=None):
        pool_counter = 0
        topo_layers = create_topo_layers(layer_collection.values())
        for i in topo_layers:
            layer = layer_collection[i]
            if isinstance(layer, InputLayer):
                keras_layer = Input(shape=(layer.shape_height, layer.shape_width, 1), name=str(layer.id))

            elif isinstance(layer, PoolingLayer):
                pool_counter += 1
                keras_layer = Lambda(dilation_pool, name=str(layer.id), arguments={'window_shape': (1, layer.pool_width), 'strides':
                    (1, layer.stride_width), 'dilation_rate': (1, 1), 'pooling_type': layer.mode})

            elif isinstance(layer, ConvLayer):
                if(layer.kernel_width == 'down_to_one'):
                    topo_layers = create_topo_layers(layer_collection.values())
                    before_conv_layer_id = topo_layers[(np.where(np.array(topo_layers) == layer.id)[0] - 1)[0]]
                    layer.kernel_width = int(layer_collection[before_conv_layer_id].keras_layer.shape[2])
                    layer.kernel_height = int(layer_collection[before_conv_layer_id].keras_layer.shape[1])
                keras_layer = Conv2D(filters=layer.filter_num, kernel_size=(layer.kernel_height, layer.kernel_width),
                                     strides=(1, 1), activation='elu', name=str(layer.id))

            elif isinstance(layer, CroppingLayer):
                keras_layer = Cropping2D(((layer.height_crop_top, layer.height_crop_bottom),
                                          (layer.width_crop_left, layer.width_crop_right)), name=str(layer.id))

            elif isinstance(layer, ZeroPadLayer):
                keras_layer = ZeroPadding2D(((layer.height_pad_top, layer.height_pad_bottom),
                                             (layer.width_pad_left, layer.width_pad_right)), name=str(layer.id))

            elif isinstance(layer, ConcatLayer):
                keras_layer = Concatenate(name=str(layer.id))(
                    [layer_collection[layer.first_layer_index].keras_layer,
                     layer_collection[layer.second_layer_index].keras_layer])

            elif isinstance(layer, BatchNormLayer):
                keras_layer = BatchNormalization(axis=layer.axis, momentum=layer.momentum, epsilon=layer.epsilon, name=str(layer.id))

            elif isinstance(layer, ActivationLayer):
                keras_layer = Activation(layer.activation_type, name=str(layer.id))

            elif isinstance(layer, DropoutLayer):
                keras_layer = Dropout(layer.rate, name=str(layer.id))

            elif isinstance(layer, FlattenLayer):
                keras_layer = Flatten(name=str(layer.id))

            elif isinstance(layer, LambdaLayer):
                keras_layer = Lambda(layer.function, name=str(layer.id))

            layer.keras_layer = keras_layer
            if layer.name is not None:
                layer.keras_layer.name = layer.name

            if layer.parent is not None:
                try:
                    layer.keras_layer = layer.keras_layer(layer.parent.keras_layer)
                except TypeError as e:
                    print('couldnt connect a keras layer with tensor...error was:', e)
                except ValueError as e:
                    print('couldnt connect a keras layer with tensor...error was:', e)
        model = MyModel(model=Model(inputs=layer_collection[next(iter(layer_collection))].keras_layer, outputs=layer.keras_layer),
                        layer_collection=layer_collection, name=name)
        model.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        try:
            assert(len(model.model.layers) == len(layer_collection) == len(topo_layers))
        except AssertionError:
            print('what happened now..?')
        return model

