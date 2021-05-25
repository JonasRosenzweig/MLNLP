# Applied Natural Language Processing with Python: Implementing Machine Learning
# and Deep Learning Algorithms for Natural Language Processing, by Taweh Beysolow II
# Chapter 1 - Keras code example

from keras import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Conv3D


def create_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size = (3,3),
                         input_shape=(None, 40, 40, 1),
                         padding='same',
                         return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same',
                         return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same',
                         return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same',
                         return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(3,3,3),
                     activation='sigmoid',
                     padding='same',
                     data_format='channels_last'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model



