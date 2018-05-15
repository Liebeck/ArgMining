"""
The CNN architectures are based on
 https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
and
"""
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
import numpy as np


def embedding_cnn(number_of_classes, index_to_embedding_mapping, padding_length,
                  kernel_size=3, dropout=0.2, filters=250):
    embedding_dimension = len(index_to_embedding_mapping[0])
    number_of_words = len(index_to_embedding_mapping.keys())
    embedding_matrix = np.zeros((number_of_words, embedding_dimension))
    for index, embedding in index_to_embedding_mapping.items():
        embedding_matrix[index] = embedding
    model = Sequential()
    embedding_layer = Embedding(number_of_words,
                                embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=padding_length)
    model.add(embedding_layer)
    model.add(Dropout(dropout))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def embedding_cnn_lstm(number_of_classes, index_to_embedding_mapping, padding_length,
                       kernel_size=3, dropout=0.2, filters=250, pool_size=4, lstm_size_layer=70):
    embedding_dimension = len(index_to_embedding_mapping[0])
    number_of_words = len(index_to_embedding_mapping.keys())
    embedding_matrix = np.zeros((number_of_words, embedding_dimension))
    for index, embedding in index_to_embedding_mapping.items():
        embedding_matrix[index] = embedding
    model = Sequential()
    embedding_layer = Embedding(number_of_words,
                                embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=padding_length)
    model.add(embedding_layer)
    model.add(Dropout(dropout))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_size_layer))
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
