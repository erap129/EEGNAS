from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout, Input
from keras.utils import to_categorical
import copy
from simanneal import Annealer
import time
from keras.callbacks import EarlyStopping

import random

class MyModel(Model):
    def __init__(self, structure=[], *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.structure = structure

    def new_model_from_structure(self, structure, naiveNAS):
        new_model = naiveNAS.base_model()
        x = new_model.layers[-1].output
        for layer in structure[4:]:
            layer_desc = layer.split('-')
            if layer_desc[0] == 'maxpool':
                x = MaxPool2D(pool_size=(1, int(layer_desc[1])), strides=(1, 3))(x)
            elif layer_desc[0] == 'convolution':
                x = Conv2D(filters=int(layer_desc[1]), kernel_size=(1, int(layer_desc[2])),
                       strides=(1, 1), activation='elu', name='convolution')(x)
        return MyModel(structure=structure, inputs=new_model.layers[0].input, output=x)


class NaiveNAS:
    def __init__(self, n_classes, input_time_len, n_chans,
                 X_train, y_train, X_valid, y_valid, X_test, y_test):
        self.n_classes = n_classes
        self.n_chans = n_chans
        self.input_time_len = input_time_len
        self.X_train = X_train
        self.X_test = X_test
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_test = y_test
        self.y_valid = y_valid

    def find_best_model(self, time_limit = 1 * 60 * 60):
        curr_model = self.base_model()
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=3)
        start_time = time.time()
        curr_acc = 0
        operations = [self.add_conv_maxpool_block, self.add_filters]
        while time.time()-start_time < time_limit:
            op_index = random.randint(0, len(operations) - 1)
            model = operations[op_index](curr_model)
            model = self.finalize_model(model)
            model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                      callbacks=[earlystopping])
            res = model.evaluate(self.X_test, self.y_test) * 100
            if res[1] >= curr_acc:
                curr_model = model
            else:
                rand = random.randint(0,1)
                if rand == 1:
                    curr_model = model
            print('model accuracy:', res[1] * 100)


    def base_model(self, n_filters_time=25, n_filters_spat=25, filter_time_length=10):
        inputs = Input(shape=(self.n_chans, self.input_time_len, 1))
        temporal_conv = Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(self.n_chans, self.input_time_len, 1),
                         kernel_size=(1, filter_time_length), strides=(1,1), activation='elu')(inputs)

        # note that this is a different implementation from the paper!
        # they didn't put an activation function between the first two convolutions

        # Also, in the paper they implemented batch-norm before each non-linearity - which I didn't do!

        # Also, they added dropout for each input the conv layers except the first! I dropped out only in the end

        spatial_conv = Conv2D(name='spatial_convolution', filters=n_filters_spat,
                              kernel_size=(self.n_chans, 1), strides=(1,1), activation='elu')(temporal_conv)
        maxpool = MaxPool2D(pool_size=(1,3), strides=(1,3))(spatial_conv)
        model = MyModel(inputs=inputs, outputs=maxpool)
        model.structure.clear()
        model.structure.append('inputs')
        model.structure.append('convolution-'+str(n_filters_time)+'-?')
        model.structure.append('convolution-'+str(n_filters_spat)+'-?')
        model.structure.append('maxpool')
        return model

    def finalize_model(self, model):
        output = model.layers[-1].output
        flatten_layer = Flatten()(output)
        prediction_layer = Dense(self.n_classes, activation='softmax', name='prediction_layer')(flatten_layer)
        model = MyModel(structure=model.structure, input=model.layers[0].input, output=prediction_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def add_conv_maxpool_block(self, model):
        conv_width = random.randint(5, 10)
        conv_filter_num = random.randint(50, 100)
        maxpool_len = random.randint(3, 5)

        output = model.layers[-1].output
        conv_layer = Conv2D(filters=conv_filter_num, kernel_size=(1, conv_width),
                            strides=(1, 1), activation='elu', name='convolution')(output)
        maxpool_layer = MaxPool2D(pool_size=(1, maxpool_len), strides=(1, 3))(conv_layer)
        model = MyModel(structure=model.structure, input=model.layers[0].input, output=maxpool_layer)
        model.structure.append('convolution-'+str(conv_filter_num)+'-'+str(conv_width))
        model.structure.append('maxpool-'+str(maxpool_len))
        return model

    # def add_skip_connection(self, model):
    #     conv_indices = [i for i, layer in enumerate(model.layers) if layer.get_config()['class_name']=='Conv2D']
    #     if len(conv_indices) < 2:
    #         return
    #     else:

    # def add_filters(self, model):
    #     conv_indices = [i for i, layer in enumerate(model.layers) if 'convolution' in layer.get_config()['name']]
    #     random_conv_index = random.randint(0, len(conv_indices)-1)
    #     factor = 2
    #     conv_layer = model.layers[random_conv_index]
    #     conv_layer.filters = conv_layer.filters * factor
    #     print('new conv layer filters after transform is:', conv_layer.filters)
    #     print('just to make sure, its:', model.layers[random_conv_index].filters)
    #     return model

    def add_filters(self, model):
        conv_indices = [i for i, layer in enumerate(model.structure) if 'convolution' in layer]
        random_conv_index = random.randint(2, len(conv_indices) - 1)
        print('layer to widen is:', str(random_conv_index))
        factor = 2
        convolution, depth, width = model.structure[conv_indices[random_conv_index]].split('-')
        model.structure[conv_indices[random_conv_index]] = convolution+'-'+str(int(depth)*factor)+'-'+width
        return model.new_model_from_structure(copy.deepcopy(model.structure), self)

