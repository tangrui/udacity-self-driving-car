from keras import backend as K
from keras.layers import Cropping2D, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.regularizers import l2
import matplotlib
matplotlib.use('Agg') # must import before pyplot
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

import data

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 10
TRAIN_BATCH_PER_EPOCH = 500
VALIDATION_BATCH_PER_EPOCH = 150

def load_model(name):
    path = './{}-model.h5'.format(name)
    if os.path.exists(path):
        print('Loading existed model: {}.', path)
        model = load_model(path)
        model.summary()
        return model

def nvidia_model():
    name = 'nvidia'
    #model = load_model(name)
    #if model is not None:
    #    return model, name

    print('Create new NVIDIA model.')
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(data.IMG_ROWS, data.IMG_COLS, data.IMG_CHS)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    #model.add(Dropout(0.5))
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    #model.add(Dropout(0.5))
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    #model.add(Dropout(0.5))
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))
    model.summary()
    return model, name

def plot_loss_curve(name, train_loss, valid_loss):
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('{}-loss-curve.png'.format(name))

def train_model(name, model, x_train, y_train):
    model.compile(loss='mse', optimizer=Adam(LEARNING_RATE))

    count = 1
    train_loss = []
    valid_loss = []
    while True:
        bias = 1.0 / count
        print('Run {}/{} (bias {:.3f})'.format(count, EPOCHS, bias))

        history = model.fit_generator(
            generator=data.train_data_generator(x_train, bias, BATCH_SIZE),
            samples_per_epoch=TRAIN_BATCH_PER_EPOCH * BATCH_SIZE,
            nb_epoch=1,
            validation_data=data.validation_data_generator(y_train, BATCH_SIZE),
            nb_val_samples=VALIDATION_BATCH_PER_EPOCH * BATCH_SIZE,
            verbose=1)
        train_loss.extend(history.history['loss'])
        valid_loss.extend(history.history['val_loss'])

        model_file_name = './{}-model-{}.h5'.format(name, count)
        print('Saving model to {}.'.format(model_file_name))
        model.save(model_file_name)

        if count == EPOCHS:
            plot_loss_curve(name, train_loss, valid_loss)
            break

        count += 1

if __name__ == '__main__':
    samples = data.read_samples()
    train_samples, valid_samples = train_test_split(samples, test_size=0.3)
    with K.get_session():
        model, name = nvidia_model()
        train_model(name, model, train_samples, valid_samples)
    exit(0)
