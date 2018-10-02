from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout
import tensorflow as tf

def deep_model(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, n_filters_5=400, filter_len_5=4):
    model = Sequential()
    model.add(Conv2D(filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1), activation='elu'))
    model.add(Conv2D(filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1,1), activation='elu'))
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