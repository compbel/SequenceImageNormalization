import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, GlobalMaxPooling1D,GlobalAveragePooling1D

import constants as cs
import derived_features as df

# fix random seed for reproducibility
np.random.seed(7)


MAX_FEATURE_LENGTH = 1000
EMBEDDING_VECTOR_LENGTH = 10
MAX_NUM_OF_FEATURES=None

def get_CNN():
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(MAX_NUM_OF_FEATURES, EMBEDDING_VECTOR_LENGTH, input_length=MAX_FEATURE_LENGTH))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters=1000,
                     kernel_size=7,
                     padding='valid',
                     activation='relu',
                     strides=1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(Dense(250))
    model.add(Dropout(0.2))

    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def get_LSTM():
    model = Sequential()
    model.add(Embedding(MAX_NUM_OF_FEATURES, EMBEDDING_VECTOR_LENGTH, input_length=MAX_FEATURE_LENGTH))

    model.add(LSTM(1, dropout=0.2))
    #model.add(Dense(250))
    #model.add(Dropout(0.2))
    #model.add(Dense(150))
    #model.add(Dropout(0.2))

    #model.add(Dense(100, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_model() :
    global MAX_NUM_OF_FEATURES, MAX_FEATURE_LENGTH

    X_train, X_test, y_train, y_test, mapping_dict = df.load_derived_data()

    MAX_NUM_OF_FEATURES=len(mapping_dict.keys())
    MAX_FEATURE_LENGTH = MAX_NUM_OF_FEATURES

    # truncate and pad input sequences
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_FEATURE_LENGTH)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_FEATURE_LENGTH)
    # create the model
    #model=get_CNN()
    model=get_LSTM()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("Accuracy: %.2f%%" % (scores[0]*100))

if __name__ == '__main__':
    train_model()