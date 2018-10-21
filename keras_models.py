from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout, BatchNormalization
from keras.models import model_from_json, Model
import numpy as np
import keras.backend as K


def shallow_model_mimic(n_chans, input_time_length, n_classes, n_filters_time=40, n_filters_spat=40,
                        filter_time_length=25, cropped=False):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1, 1)))
    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Lambda(lambda x: x ** 2))  # squaring layer
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 75), 'strides': (1, 15), 'dilation_rate': (1, 1), 'pooling_type': 'AVG'}))
    model.add(Lambda(lambda x: K.log(K.clip(x, min_value=1e-6, max_value=None))))
    model.add(Conv2D(filters=n_classes, kernel_size=(1, 69), strides=(1, 1)))
    model.add(Dropout(0.5))
    if cropped:
        model.add(Lambda(mean_layer))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def dilation_pool(x, window_shape, strides, dilation_rate, pooling_type='MAX'):
    import tensorflow as tf
    return tf.nn.pool(x, window_shape=window_shape, strides=strides, dilation_rate=dilation_rate,
                      pooling_type=pooling_type, padding='VALID')


def mean_layer(x):
    import keras.backend as K
    return K.mean(x, axis=2)


# trying to mimic exactly what was done in pytorch in the paper implementation
def deep_model_mimic(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, cropped=False):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1)))
    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1,1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    # model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1,3), 'strides': (1,3), 'dilation_rate': (1,1)}))

    model.add(Conv2D(filters=n_filters_2, kernel_size=(1, filter_len_2), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    # model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 3), 'strides': (1, 3), 'dilation_rate': (1,1)}))

    model.add(Conv2D(filters=n_filters_3, kernel_size=(1, filter_len_3), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    # model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 3), 'strides': (1, 3), 'dilation_rate': (1,1)}))

    model.add(Dropout(0.5))
    model.add(Conv2D(filters=n_filters_4, kernel_size=(1, filter_len_4), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    # model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 3), 'strides': (1, 3), 'dilation_rate': (1,1)}))

    final_kernel_size = int(model.layers[-1].output_shape[2])
    model.add(Conv2D(filters=n_classes, kernel_size=(1, final_kernel_size), strides=(1, 1)))
    model.add(Activation('softmax'))
    if cropped:
        model.add(Lambda(mean_layer))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def convert_to_dilated(model):
    axis = [0,1]
    stride_so_far = np.array([1,1])
    for layer in model.layers:
        if hasattr(layer, 'dilation_rate') or (hasattr(layer, 'arguments') and 'dilation_rate' in layer.arguments):
            if hasattr(layer, 'arguments'):
                dilation_rate = layer.arguments['dilation_rate']
            else:
                dilation_rate = layer.dilation_rate
            assert dilation_rate == 1 or (dilation_rate == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?")
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            if hasattr(layer, 'arguments'):
                layer.arguments['dilation_rate'] = tuple(new_dilation)
            else:
                layer.dilation_rate = tuple(new_dilation)
        if hasattr(layer, 'strides') or (hasattr(layer, 'arguments') and 'strides' in layer.arguments):
            if hasattr(layer, 'arguments'):
                strides = layer.arguments['strides']
            else:
                strides = layer.strides
            if not hasattr(strides, '__len__'):
                strides = (strides, strides)
            stride_so_far *= np.array(strides)
            new_stride = list(strides)
            for ax in axis:
                new_stride[ax] = 1
            if hasattr(layer, 'arguments'):
                layer.arguments['strides'] = tuple(new_stride)
            else:
                layer.strides = tuple(new_stride)
    new_model = model_from_json(model.to_json())
    print(model.layers)
    new_model.summary()
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model



def deep_model_cropped(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, n_filters_5=400, filter_len_5=4):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1), activation='elu'))
    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1,1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_2, kernel_size=(1, filter_len_2), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_3, kernel_size=(1, filter_len_3), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Conv2D(filters=n_filters_4, kernel_size=(1, filter_len_4), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Dropout(0.5))

    model.add(Conv2D(filters=n_classes, kernel_size=(1, 2), strides=(1, 1), activation='softmax'))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model




def deep_model(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, n_filters_5=400, filter_len_5=4):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1), activation='elu'))

    # note that this is a different implementation from the paper!
    # they didn't put an activation function between the first two convolutions
    # Also, in the paper they implemented batch-norm before each non-linearity - which I didn't do!
    # Also, they added dropout for each input the conv layers except the first! I dropped out only in the end

    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1,1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_2, kernel_size=(1, filter_len_2), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_3, kernel_size=(1, filter_len_3), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Conv2D(filters=n_filters_4, kernel_size=(1, filter_len_4), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Dropout(0.5))

    model.add(Conv2D(filters=n_filters_5, kernel_size=(1, filter_len_5), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Dropout(0.5))

    model.add(Conv2D(filters=n_classes, kernel_size=(1, 2), strides=(1, 1), activation='softmax'))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

