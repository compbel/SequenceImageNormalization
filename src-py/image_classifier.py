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
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os, glob, shutil
import constants as cs

# dimensions of our images.
img_width, img_height = 480, 480

top_model_weights_path = 'bottleneck_fc_model.h5'
data_dir = cs.IMAGE_DATA_DIR
nb_train_samples = 292
nb_validation_samples = 73
epochs = 50
batch_size = 16
test_size = 0.2

'''
Generates training and validation data for images in the following dir structure.
Please refer to constants.py for exact dir paths/names
```
../data/img_data
    train/
        Chronics_NGS/
            Chronics_NGS_plot_01_.png
            ...
        Acutes_NGS2/
            Acutes_NGS2_plot_01_.png
            ...
    test/
        Chronics_NGS/
            Chronics_NGS_plot_01_.png
            ...
        Acutes_NGS2/
            Acutes_NGS2_plot_01_.png
            ...
```
'''
def gen_train_test_data():
    chr_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CHRONIC_DIR_NAME + "*")
    clinic_files = glob.glob(cs.IMAGE_DATA_DIR_ALL + "*" + cs.CLINIC_DIR_NAME + "*")

    data = list(chr_files)
    data.extend(clinic_files)
    data_label = [cs.CHRONIC_LABEL] * len(chr_files)
    data_label.extend([cs.CLINIC_LABEL] * len(clinic_files))

    img_train, img_test, y_train, y_test = train_test_split(data, data_label, test_size=test_size, random_state=42)

    # Persist files
    # Remove exisitng files from dirs
    try:
        shutil.rmtree(cs.IMAGE_DATA_DIR_TRAIN)
        shutil.rmtree(cs.IMAGE_DATA_DIR_TEST)
    except:
        print "No dir to delete"

    # Create dirs
    dirs = [cs.IMAGE_DATA_DIR_TRAIN + cs.CHRONIC_DIR_NAME, cs.IMAGE_DATA_DIR_TEST + cs.CHRONIC_DIR_NAME,
            cs.IMAGE_DATA_DIR_TRAIN + cs.CLINIC_DIR_NAME, cs.IMAGE_DATA_DIR_TEST + cs.CLINIC_DIR_NAME]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # copy train label files - Chronic
    for filename in img_train:
        if cs.CHRONIC_DIR_NAME in filename:
            shutil.copy(filename, cs.IMAGE_DATA_DIR_TRAIN + cs.CHRONIC_DIR_NAME)
        else:
            shutil.copy(filename, cs.IMAGE_DATA_DIR_TRAIN + cs.CLINIC_DIR_NAME)

    # copy train label files - Chronic
    for filename in img_test:
        if cs.CHRONIC_DIR_NAME in filename:
            shutil.copy(filename, cs.IMAGE_DATA_DIR_TEST + cs.CHRONIC_DIR_NAME)
        else:
            shutil.copy(filename, cs.IMAGE_DATA_DIR_TEST + cs.CLINIC_DIR_NAME)

    print "Train: ", len(img_train)
    print "Test: ", len(img_test)

    return img_train, img_test, y_train, y_test

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=( img_height, img_width,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

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
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model;


def save_bottlebeck_features(train_data_dir, validation_data_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    #model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    # bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples // batch_size)
    # np.save(open(train_data_dir + 'bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    test_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    # bottleneck_features_validation = model.predict_generator( generator, nb_validation_samples // batch_size)
    # np.save(open(validation_data_dir + 'bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    return train_generator, test_generator


def train_top_model(train_data_dir, validation_data_dir):
    '''
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    '''
    train_generator, validation_generator = save_bottlebeck_features(train_data_dir, validation_data_dir)

    model = get_model((480, 480, 3))

    model.fit_generator(train_generator, epochs=epochs, batch_size=batch_size, validation_data=validation_generator)
    model.save_weights(top_model_weights_path)


def test_model(train_data_dir, validation_data_dir):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                        batch_size=32, class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
    model = get_conv_model();

    model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=70)


if __name__ == '__main__':
    train_data_dir =cs.IMAGE_DATA_DIR_TRAIN
    validation_data_dir=cs.IMAGE_DATA_DIR_TEST
    # gen_train_test_data()

    train_top_model(train_data_dir, validation_data_dir)
    test_model(train_data_dir, validation_data_dir)

