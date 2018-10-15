from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import random


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

    def find_best_model(self):
        base_model = self.base_model()
        model = self.add_conv_maxpool_block(base_model)
        model = self.add_filters(model)
        model = self.finalize_model(model)
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10)
        model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                  callbacks=[earlystopping])
        res = model.evaluate(self.X_test, self.y_test) * 100
        print('model accuracy:', res[1] * 100)

    def base_model(self, n_filters_time=25, n_filters_spat=25, filter_time_length=10):
        model = Sequential()
        model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(self.n_chans, self.input_time_len, 1),
                         kernel_size=(1, filter_time_length), strides=(1,1), activation='elu'))

        # note that this is a different implementation from the paper!
        # they didn't put an activation function between the first two convolutions

        # Also, in the paper they implemented batch-norm before each non-linearity - which I didn't do!

        # Also, they added dropout for each input the conv layers except the first! I dropped out only in the end

        model.add(Conv2D(name='spatial_convolution', filters=n_filters_spat, kernel_size=(self.n_chans, 1), strides=(1,1), activation='elu'))
        model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))
        return model

    def finalize_model(self, model):
        output = model.layers[-1].output
        flatten_layer = Flatten()(output)
        prediction_layer = Dense(self.n_classes, activation='softmax', name='prediction_layer')(flatten_layer)
        model = Model(input=model.layers[0].input, output=prediction_layer)
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
        model = Model(input=model.layers[0].input, output=maxpool_layer)
        return model

    # def add_skip_connection(self, model):
    #     conv_indices = [i for i, layer in enumerate(model.layers) if layer.get_config()['class_name']=='Conv2D']
    #     if len(conv_indices) < 2:
    #         return
    #     else:

    def add_filters(self, model):
        conv_indices = [i for i, layer in enumerate(model.layers) if 'convolution' in layer.get_config()['name']]
        random_conv_index = random.randint(0, len(conv_indices)-1)
        factor = 2
        conv_layer = model.layers[random_conv_index]
        conv_layer.filters = conv_layer.filters * factor
        print('new conv layer filters after transform is:', conv_layer.filters)
        print('just to make sure, its:', model.layers[random_conv_index].filters)
        return model




