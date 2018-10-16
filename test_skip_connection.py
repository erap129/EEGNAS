from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout, Concatenate, Cropping2D


def skip_model(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, n_filters_5=400, filter_len_5=4):
    inputs = Input(shape=(n_chans, input_time_length, 1))
    temp_conv = Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1), activation='elu')(inputs)

    # note that this is a different implementation from the paper!
    # they didn't put an activation function between the first two convolutions

    # Also, in the paper they implemented batch-norm before each non-linearity - which I didn't do!

    # Also, they added dropout for each input the conv layers except the first! I dropped out only in the end

    spat_conv = Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1),
                       strides=(1,1), activation='elu')(temp_conv)
    maxpool = MaxPool2D(pool_size=(1,3), strides=(1,3))(spat_conv)

    conv1 = Conv2D(filters=n_filters_2, kernel_size=(1, filter_len_2), strides=(1, 1), activation='elu')(maxpool)
    maxpool1 = MaxPool2D(pool_size=(1,3), strides=(1,3))(conv1)

    # crop_conv = Cropping2D(cropping=((0, 0), (121, 121)))(conv1)
    crop_conv = Cropping2D(cropping=((0, 0), (-20, -20)))(conv1)
    concat = Concatenate()([crop_conv, maxpool1])
    conv2 = Conv2D(filters=n_filters_3, kernel_size=(1, filter_len_3), strides=(1, 1), activation='elu')(concat)
    maxpool2 = MaxPool2D(pool_size=(1, 3), strides=(1, 3))(conv2)

    conv3 = Conv2D(filters=n_filters_4, kernel_size=(1, filter_len_4), strides=(1, 1), activation='elu')(maxpool2)
    maxpool3 = MaxPool2D(pool_size=(1, 3), strides=(1, 3))(conv3)

    dropout = Dropout(0.5)(maxpool3)

    conv4 = Conv2D(filters=n_filters_5, kernel_size=(1, filter_len_5), strides=(1, 1), activation='elu')(dropout)
    maxpool4 = MaxPool2D(pool_size=(1, 3), strides=(1, 3))(conv4)

    dropout = Dropout(0.5)(maxpool4)

    final_conv = Conv2D(filters=n_classes, kernel_size=(1, 2), strides=(1, 1), activation='softmax')(dropout)
    flatten = Flatten()(final_conv)
    model = Model(inputs=inputs, outputs=flatten)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

