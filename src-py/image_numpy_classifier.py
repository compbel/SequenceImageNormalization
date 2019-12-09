'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:

'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import applications
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import constants as cs
import image_data_utils as idu

# dimensions of our images.
img_width, img_height = 480, 480

top_model_weights_path = 'bottleneck_fc_model.h5'
data_dir = cs.IMAGE_DATA_DIR
nb_train_samples = 292
nb_validation_samples = 73
epochs = 50
batch_size = 16
test_size = 0.2


def get_conv_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model(input_shape):
    model = Sequential()
    #model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model;

def test_model():
    x_train, x_test, y_train, y_test = idu.load_img_numpy_data()
    print "Shape: ", x_train.shape[1:]

    model = get_conv_model(x_train.shape[1:]);
    #model = get_model(x_train.shape[1:]);

    model.fit(x_train, y_train,
              epochs=50,
              batch_size=batch_size,
              validation_data=(x_test, y_test))
    model.save_weights(cs.IMAGE_DATA_DIR+'bottleneck_fc_model.h5')

if __name__ == '__main__':
    train_data_dir =cs.IMAGE_DATA_DIR_TRAIN
    validation_data_dir=cs.IMAGE_DATA_DIR_TEST

    #persist_img_numpy_data()

    test_model()

